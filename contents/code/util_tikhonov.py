"""
DRT from EIS
Distribution of Relaxation Time analyzer using Electrochemical Impedance Spectrum Data

DRTSettings: settings for DRTAnalyzer
    Sample code:
        drt_settings = DRTSettings()
        drt_settings.lmd = 1e-3  # change the settings in this way

DRTAnalyzer: calculates DRT from EIS
    Sample code:
        drt_analyzer = DRTAnalyzer()
        drt_analyzer.settings = drt_settings  # apply the settings
        if your data is minimally noisy:
            # Units of input EIS data: Hz, Ohm, Ohm
            L, R_inf_ref, gamma, tau, phi, w = drt_analzer.drt_tknv(eis_freq, eis_z_real, eis_z_imag, weights=myweights)
            # L = np.array([L_self, L_wire]) Henry, self induction changes eis_z_imag, wire induction changes eis_z_real
            # R_inf_ref, series resistance, Ohm
            # gamma: z_eis = integrate(from -Inf to +Inf, gamma(tau) * d(ln(tau)) / (1 + j * omega * tau))
                omega = 2 * pi * eis_freq
            # tau: relaxation times, seconds
            # weights is optional. Applied to the evaluation of EIS data. Need not to be normalized.
            # len(weights) must == len(eis_freq)
            # phi: matrix, phi @ x = gamma, x is the vector of RBF function intensities, see DRTtools at
                                                                    https://doi.org/10.1016/j.electacta.2015.09.097
            # w: weights applied on EIS data when evaluating DRT. Low weights might indicate poor EIS data reliability
        else: you data is somewhat or very noisy:
            L, R_inf_ref, gamma, tau, phi, w = drt_analyzer.drt_iterative(eis_freq, eis_z_real, eis_z_imag,
                                                                   iter_count=3, bad_point_suppress=4, err_thresh=0.015,
                                                                   trim_thresh=0.1)
            # iter_count: number of iterations to refine the result / reduce the impact of noise
            # bad_point_suppress: integers from 1 on, greater values result in faster convergence,
                but may rule out usable points, 4 as the default value is recommended
            # err_thresh: error threshold below which the iterations are SKIPPED
            # trim_thresh: error threshold over which the most erroneous part of EIS data will be trimmed

        # If you wish to convert the DRT result back to impedance spectra, you may use:
        eis_complex_impedance = drt_analyzer.eis_from_drt(L, R_inf_ref, gamma, tau, eis_freq)
        # or if you wish to calculate multiple EIS from DRT and speed up the code, you may get a transformation matrix
            and calculate yourself. Note that this method assumes pulsed DRT and thus is not mathematically accurate:
        mat = drt_analyzer.eis_from_drt_mat(tau, eis_freq)
        eis_complex_impedance = 2 * np.pi * freq * (1j * L[0] + L[1]) + R_inf_ref + mat @ gamma

Coded by Hangyue Li (Leo)
Redundence removed and speed up, fast data screening (valid_indices)
2022.01.28
Replaced auto-lambda and hyper-lambda with weight-dependent auto lambda
2021.06.08
Normalized the roughness penalty coef lmd to the number frequency points in EIS data
2021.05.27
R0 addition in drt_iterative removed, impedance modulus weighting in get_Hb improved
2021.02.26
R_inf and L can be user-specified, DRT evaluation based on the specified values
2021.02.24
Added auto-lambda algorithm according to Zhang et al. (see code comments for full reference, search 'auto_lmd'
2020.10.13
updated DRTAnalyzer.valid_indices(), faster and less mistakenly invalidated data
2020.04.30
this version picked up quadprog again, valid indices filtering method improved
2020.04.22
previous version 1.3 opted for scipy solver in place of quadprog
2020.04.09
previous version 1.2 added drt_iterative and related
2020.03.16
previous version 1.1
2019.12.29
"""

message_popup_switch = False


import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.signal import fftconvolve, convolve
# from scipy.optimize import lsq_linear
# from cvxopt.solvers import coneqp, options
# from cvxopt import matrix
from quadprog import solve_qp as qp  # this module requires C compiler
# from qpsolvers import solve_qp as qp
from copy import deepcopy
from os import path, makedirs


class BaseGaussFunc:
    def __init__(self, mu, delta):
        self.mu = mu
        self.delta = delta
        half_width = 15.0 / mu
        self.left_re = -half_width
        self.right_re = half_width
        self.left_im = -half_width
        self.right_im = half_width

    def f_re(self, x):
        ee = np.exp(x + self.delta)
        gauss = np.exp(-(self.mu * x) ** 2)
        return gauss / (1 + ee ** 2)

    def f_im(self, x):
        ee = np.exp(x + self.delta)
        gauss = np.exp(-(self.mu * x) ** 2)
        re = gauss / (1 + ee ** 2)
        return re * ee


class MatFactory:
    def __init__(self):
        libpath = path.dirname(path.abspath(__file__)) + '/libtikhonov/'
        try:
            AAre = np.load(libpath + 'AAre.npy')
            AAim = np.load(libpath + 'AAim.npy')
            mus = np.load(libpath + 'mus.npy')
            deltas = np.load(libpath + 'deltas.npy')
        except FileNotFoundError:
            if not path.exists(libpath):
                makedirs(libpath)
            AAre, AAim, mus, deltas = MatFactory.make_lib()
            np.save(libpath + 'AAre.npy', AAre)
            np.save(libpath + 'AAim.npy', AAim)
            np.save(libpath + 'mus.npy', mus)
            np.save(libpath + 'deltas.npy', deltas)
        self.AArelib = AAre
        self.AAimlib = AAim
        self.mus = mus
        self.deltaslib = deltas
        self.deltas = deltas
        self.imu = int(len(mus) / 2)  # default value: the value in the middle of mus
        self.mu = mus[self.imu]
        self.AAre = AAre[self.imu]
        self.AAim = AAim[self.imu]

    def setmu(self, mu):
        if not mu > 0:
            raise ValueError('Parameter mu must be positive')
        mus = self.mus  # Leo: I just don't want to type self.mus for mus whenever I use it
        if mus[0] <= mu <= mus[-1]:
            mu_count = len(mus)
            diff_last = np.inf  # initial value
            imu = 0  # so that imu is accessible outside the for loop
            for imu in range(mu_count):  # find the index to the closest value to mu in mus
                diff = mu - mus[imu]
                if diff > 0:  # search for the member in mus that is no less than mu
                    continue
                elif diff_last > -diff:  # the current member in mus is closer to mu
                    break  # retain the current imu
                else:
                    imu -= 1  # set imu as its last value
                    break
            self.imu = imu
            self.mu = mus[imu]
            self.AAre = self.AArelib[imu]
            self.AAim = self.AAimlib[imu]
        else:
            self.AAre, self.AAim, _, self.deltas = MatFactory.make_lib(mu_min=self.mu, mu_max=self.mu, mu_count=1)
            self.AAre = self.AAre[0]
            self.AAim = self.AAim[0]
            self.mu = mu
        return self.mu

    def calcM(self, delta):  # x @ M @ x = Integrate((d gamma / d lntau) ^ 2, -inf < lntau < inf)
        mudel2 = (self.mu * delta) ** 2
        ee = np.exp(-0.5 * mudel2)  # Integral of first-order derivative of gamma(lntau)
        return np.sqrt(np.pi / 2) * ee * self.mu * (1 - mudel2)

    def calcN(self, delta):  # x @ N @ x = Integrate((d^2 gamma / d lntau^2) ^ 2, -inf < lntau < inf)
        mudel2 = (self.mu * delta) ** 2
        ee = np.exp(-0.5 * mudel2)  # Integral of second-order derivative of gamma(lntau)
        return np.sqrt(np.pi / 2) * ee * self.mu ** 3 * (3 - 6 * mudel2 + mudel2 ** 2)

    def deltaM_density(self, x):  # the sampling density function of deltaM
        return 1 - 0.995 / ((self.mu * x / 3) ** -2 + 1)

    def getAMP(self, freqs, tau_drt, tau_out, inf_mu=False):  # see ref [1] for the meaning of matrix A and M
        freq_eis = np.array(freqs)
        freq_drt = 1 / (2 * np.pi * np.array(tau_drt))
        freq_phi = 1 / (2 * np.pi * np.array(tau_out))
        l_eis = len(freq_eis)
        l_drt = len(freq_drt)
        l_phi = len(freq_phi)
        delta_matrixA = np.log(np.array([freq_eis] * l_drt)).T - np.log(np.array([freq_drt] * l_eis))
        delta_matrixM = np.log(np.array([freq_drt] * l_drt)).T - np.log(np.array([freq_drt] * l_drt))
        delta_matrixP = np.log(np.array([freq_phi] * l_drt)).T - np.log(np.array([freq_drt] * l_phi))
        delta_min = min(np.min(delta_matrixA), np.min(delta_matrixM))
        delta_max = max(np.max(delta_matrixA), np.max(delta_matrixM))
        if delta_min < np.min(self.deltaslib) or delta_max > np.max(self.deltaslib):  # required deltas exceed lib data
            if delta_min != np.min(self.deltas) or delta_max != np.max(self.deltas):
                # required deltas different from last time
                self.AAre, self.AAim, _, self.deltas = MatFactory.make_lib(mu_min=self.mu, mu_max=self.mu, mu_count=1,
                                                                           delta_min=delta_min, delta_max=delta_max)
                self.AAre = self.AAre[0]
                self.AAim = self.AAim[0]
        else:
            self.AAre = self.AArelib[self.imu]
            self.AAim = self.AAimlib[self.imu]

        safety_margin = 0.1
        deltasM = MatFactory.make_spacing(self.deltaM_density,
                                          delta_min - safety_margin, delta_max + safety_margin, 1000)
        deltasP = MatFactory.make_spacing(self.deltaM_density, np.min(delta_matrixP) - safety_margin,
                                          np.max(delta_matrixP) + safety_margin, 1000)
        MM = self.calcM(deltasM)
        NN = self.calcN(deltasM)
        LNPHI = -(self.mu * deltasP) ** 2
        DPHI = -2 * deltasM * self.mu ** 2 * np.exp(-(self.mu * deltasM) ** 2)
        D2PHI = -2 * self.mu ** 2 * (1 - 2 * (self.mu * deltasM) ** 2) * np.exp(-(self.mu * deltasM) ** 2)

        # use piece-wise linear interpolation to avoid overshoot
        if inf_mu:
            AA_infmu = 1 / (1 + 1j * np.exp(self.deltas))
            cs = interp1d(self.deltas, np.log(np.maximum(AA_infmu.real, 1e-308)))
            Are = np.exp(cs(delta_matrixA))
            cs = interp1d(self.deltas, np.log(np.maximum(-AA_infmu.imag, 1e-308)))
            Aim = -np.exp(cs(delta_matrixA))
            M = delta_matrixM * 0.0
            N = delta_matrixM * 0.0
            phi = delta_matrixP * 0.0
            for j, col in enumerate(delta_matrixP.T):
                bias = np.abs(col)
                i = int(np.argmin(bias))
                pulse_height = 1.0 / np.abs(np.log(freq_phi[i] / freq_phi[i - 1]))
                if i == 0:
                    w1, w2 = self._weights(0.0, col[i], col[i + 1])
                    phi[i][j] = pulse_height * w1
                    phi[i + 1][j] = pulse_height * w2
                elif i == len(freq_phi) - 1:
                    w1, w2 = self._weights(0.0, col[i - 1], col[i])
                    phi[i - 1][j] = pulse_height * w1
                    phi[i][j] = pulse_height * w2
                else:
                    if bias[i - 1] < bias[i + 1]:
                        w1, w2 = self._weights(0.0, col[i - 1], col[i])
                        phi[i - 1][j] = pulse_height * w1
                        phi[i][j] = pulse_height * w2
                    else:
                        w1, w2 = self._weights(0.0, col[i], col[i + 1])
                        phi[i][j] = pulse_height * w1
                        phi[i + 1][j] = pulse_height * w2
            dphi = delta_matrixM * 0.0
            d2phi = delta_matrixM * 0.0
            L = delta_matrixM * 0.0
        else:
            cs = interp1d(self.deltas, np.log(np.maximum(self.AAre, 1e-308)))
            Are = np.exp(cs(delta_matrixA))
            cs = interp1d(self.deltas, np.log(np.maximum(-self.AAim, 1e-308)))
            Aim = -np.exp(cs(delta_matrixA))
            cs = interp1d(deltasM, MM)
            M = cs(delta_matrixM)
            cs = interp1d(deltasM, NN)
            N = cs(delta_matrixM)
            cs = interp1d(deltasP, LNPHI)
            phi = np.exp(cs(delta_matrixP))
            cs = interp1d(deltasM, DPHI)
            dphi = cs(delta_matrixM)
            cs = interp1d(deltasM, D2PHI)
            d2phi = cs(delta_matrixM)
            L = -2 * self.mu ** 2 * delta_matrixM * np.exp(- self.mu ** 2 * delta_matrixM ** 2)
        return Are, Aim, M, phi, [N, dphi, d2phi, L]

    @staticmethod
    def _weights(x, left, right, tol=1e-6):  # the tolerance is this large because w1, w2 are only used for plotting
        rel_pos = (x - left) / (right - left)
        if not 0 - tol <= rel_pos <= 1 + tol:
            raise ValueError('The input arguments must satisfy: left <= x <= right')
        rel_pos = max(0, min(rel_pos, 1))
        w1 = 1.0 - rel_pos
        w2 = rel_pos
        return w1, w2

    @staticmethod
    def make_spacing(density, min_val, max_val, count):
        if callable(density):  # then map it into an array
            delta_linspace = np.linspace(min_val, max_val, count)
            density = np.array(list(map(density, delta_linspace)))
        elif hasattr(density, '__len__') and len(density) != count:  # an array
            if len(density) > 1:
                interplin = interp1d(np.linspace(0, 1, len(density)),
                                     density)  # piece-wise linear interpolation
                density = interplin(np.linspace(0, 1, count))
            else:
                density = np.ones(count)
        if np.min(density) < 1.0e-307:  # to avoid overflow
            raise ValueError('delta_density contains zeros or negative values')
        cumsum = np.cumsum(density)
        cumsum = (cumsum - cumsum[0]) / (cumsum[-1] - cumsum[0])  # normalize
        delta_nlinsp = np.linspace(0, 1, count)  # normal linear spacing (from 0 to 1)
        interplin = interp1d(cumsum, delta_nlinsp)  # piece-wise linear interpolation, invert function
        x = interplin(delta_nlinsp)
        x = x * (max_val - min_val) + min_val
        return x

    @staticmethod
    def make_lib(mu_min=0.5, mu_max=50.0, mu_count=101, delta_min=-50.0, delta_max=50.0, delta_count=1001,
                 delta_density=tuple([])):
        if message_popup_switch:
            from tkinter import messagebox
            messagebox.showwarning(title=None, message=None)
        else:
            print('Generating library... This may take a few minutes.\n')
        if delta_density == tuple([]):  # default value
            def delta_density(x):
                if abs(x) < 1e-75:
                    return 1.0
                else:
                    return 1 - 0.9 / (1 / (x / 4) ** 4 + 1)
        if mu_count > 1:
            lnmu_min = np.log(mu_min)
            lnmu_max = np.log(mu_max)
            lnmus = np.linspace(lnmu_min, lnmu_max, mu_count)
            mus = np.exp(lnmus)
        else:
            mus = np.array([mu_min])
        if delta_count > 1:
            deltas = MatFactory.make_spacing(delta_density, delta_min, delta_max, delta_count)
        else:
            deltas = np.array([delta_min])
        AAre = np.zeros([mu_count, delta_count])
        AAim = np.zeros([mu_count, delta_count])
        if mus.size > 1 and deltas.size > 1:
            for imu, mu in enumerate(mus):
                for idelta, delta in enumerate(deltas):
                    bgf = BaseGaussFunc(mu, delta)
                    AAre[imu][idelta] = quad(bgf.f_re, bgf.left_re, bgf.right_re)[0]
                    AAim[imu][idelta] = -quad(bgf.f_im, bgf.left_im, bgf.right_im)[0]
        elif deltas.size > 1:
            for idelta, delta in enumerate(deltas):
                bgf = BaseGaussFunc(mus[0], delta)
                AAre[0][idelta] = quad(bgf.f_re, bgf.left_re, bgf.right_re)[0]
                AAim[0][idelta] = -quad(bgf.f_im, bgf.left_im, bgf.right_im)[0]
        else:
            bgf = BaseGaussFunc(mus[0], deltas[0])
            AAre[0][0] = quad(bgf.f_re, bgf.left_re, bgf.right_re)[0]
            AAim[0][0] = -quad(bgf.f_im, bgf.left_im, bgf.right_im)[0]
        return AAre, AAim, mus, deltas

    @staticmethod
    def getHb(Are, Aim, M, N, f_eis, z_eis_re, z_eis_im, lmd, L_penalty=(0.0, 0.0), weight_re=0.5, weights=None,
              R_inf_set=None, L_set=None):
        # lmd = lambda, regularization parameter, weight_re: 0 - 1, the balance between re & im of eis_data in loss
        # L_set = [element1, element2], element 1 / 2 can each be a double or None
        l_eis = len(Are)  # == len(A) row count == len(f_eis) == len(z_eis)
        l_drt = len(M)  # == len(A[0]) == len(M[0]) column count
        z_eis_re = np.array(z_eis_re)
        z_eis_im = np.array(z_eis_im)
        if weights is None:
            weights = np.ones(l_eis)
        else:
            weights = np.array(weights)
        # z_mod2 = z_eis_re ** 2 + z_eis_im ** 2
        # w = weights / (z_mod2 + np.average(z_mod2))  # standardized weighting
        w = weights * 1.0  # copy
        w = w / np.average(w)
        # w = np.maximum(1e-9, w)  # avoid zero weights being passed to H, so that H is spd.
        w = np.diag(w)

        if not hasattr(L_penalty, '__len__'):
            L_penalty = np.array([L_penalty, L_penalty])
        L_penalty = np.minimum(np.maximum(1, np.array(L_penalty) + 1), 1e6)

        #        [ 0                1      ]        [              0  0      ]      [ 0  0  0  0 ... 0 ]
        #        [ .                .      ]        [              .  .      ]      [ 0  0  0  0 ... 0 ]
        #        [ .                .      ]        [              .  .      ]      [ 0  0  0  0 ... 0 ]
        # Cre =  [ .  2*pi*f_eis uH .  Are ], Cim = [2*pi*f_eis uH .  .  Aim ], T = [ .  .  .          ]
        #        [ .                .      ]        [              .  .      ]      [ .  .  .     M    ]
        #        [ 0                1      ]        [              0  0      ]      [ 0  0  0          ]
        Cre = np.hstack([np.zeros([l_eis, 1]), L_penalty[1] * 2 * np.pi * np.array([f_eis]).T * 1e-6
                         * (L_set[1] is None),
                         np.ones([l_eis, 1]) * (R_inf_set is None), Are])
        Cim = np.hstack([L_penalty[0] * 2 * np.pi * np.array([f_eis]).T * 1e-6 * (L_set[0] is None),
                         np.zeros([l_eis, 2]), Aim])
        T = np.vstack([np.zeros([3, l_drt + 3]), np.hstack([np.zeros([l_drt, 3]), M])])
        T2 = np.vstack([np.zeros([3, l_drt + 3]), np.hstack([np.zeros([l_drt, 3]), N])])

        weight_re *= 2  # this is a scalar/number
        weight_im = 2 - weight_re
        lmd = np.sqrt(lmd)
        if hasattr(lmd, '__len__'):
            lmd = np.diag(np.append([0.0] * 3, lmd))
        else:
            lmd = np.diag(np.append([0.0] * 3, np.ones(l_eis) * lmd))
        if R_inf_set is not None:
            z_eis_re -= R_inf_set
        if L_set[0] is not None:
            z_eis_im -= 2 * np.pi * f_eis * L_set[0]
        if L_set[1] is not None:
            z_eis_re -= 2 * np.pi * f_eis * L_set[1]
        H = weight_re * Cre.T @ w @ Cre + weight_im * Cim.T @ w @ Cim + lmd @ (T + T2) @ lmd * (l_eis / 73)
        b = weight_re * z_eis_re.T @ w @ Cre + weight_im * z_eis_im.T @ w @ Cim
        H[0, 0] *= L_penalty[0]
        H[1, 1] *= L_penalty[1]

        return H, b  # H @ x = b, x = [L, R_inf_ref, gamma(f_drt)], L = inductance (uH), R_inf_ref = z_re(freq == inf) (Ohm)


class Loss:
    def __init__(self, H, b):
        self.H = np.array(H).real
        self.b = np.array(b).real

    def fHb(self, x):
        x = np.array(x)
        return 0.5 * (self.H @ x - 2 * self.b).T @ x

    def grad(self, x):
        x = np.array(x)
        return self.H @ x - self.b

    def hess(self, x):
        return self.H

    def hessp(self, x, p):
        p = np.array(p)
        return self.H @ p


class LossAutoLmd:  # L in uH
    def __init__(self, A, f, z, weights, L, D, fit_R_inf=True, fit_L_self=True, fit_L_wire=True):
        f_col = f.reshape([len(f), 1])
        AA = np.hstack([fit_L_self * 2e-6j * np.pi * f_col, fit_L_wire * 2e-6 * np.pi * f_col,
                        fit_R_inf * np.ones([len(f), 1]), A])
        self.AA = np.diag(weights) @ AA
        self.f = f
        self.z = z * weights
        self.L = L  # L is the penalty matrix of xm (original solution of DRT, gamma = phi @ xm, phi -> see getAMP)
        self.D = D  # D is the penalty matrix of lambda
        self.len_xm = self.L.shape[0]
        self.len_lmd = self.D.shape[0]

    def fnc(self, vec):
        vec = np.array(vec)  # vec = [L_self, L_wire, R_inf, xm, ln(lmd)], offset == len(L_self, L_wire, R_inf]) == 3
        offset = vec.size - self.len_xm - self.len_lmd
        if offset != 3:
            raise ValueError("Unexpected length of the variable vector.")
        x = vec[:self.len_xm + offset]
        xm = x[-self.len_xm:]
        lmd = np.exp(vec[-self.len_lmd:])
        fHb = self.AA @ x - self.z
        fHb = np.sum(fHb * np.conj(fHb)).real
        Lxm = self.L @ xm
        xm_penalty = np.sum((Lxm * lmd) ** 2)
        Dlmd = self.D @ lmd
        lmd_penalty = np.sum(xm ** 2) * np.sum(Dlmd ** 2)
        ret = (fHb + xm_penalty + lmd_penalty) * 0.5
        return ret

    def grad(self, vec):
        vec = np.array(vec)
        offset = vec.size - self.len_xm - self.len_lmd
        if offset != 3:
            raise ValueError("Unexpected length of the variable vector.")
        x = vec[:self.len_xm + offset]
        xm = x[-self.len_xm:]
        lmd = np.exp(vec[-self.len_lmd:])
        dfdx = np.conj(self.AA.T) @ (self.AA @ x - self.z)
        df = np.append(dfdx.real, np.zeros(self.len_lmd))
        Lxm = self.L @ xm
        dxm_penaltydxm = (Lxm * lmd ** 2) @ self.L
        dxm_penaltydlmd = (lmd * Lxm) ** 2
        dxm_penalty = np.append(dxm_penaltydxm, dxm_penaltydlmd)
        Dlmd = self.D @ lmd
        dlmd_penaltydxm = np.sum(Dlmd ** 2) * xm
        dlmd_penaltydlmd = np.sum(xm ** 2) * self.D.T @ Dlmd * lmd
        dlmd_penalty = np.append(dlmd_penaltydxm, dlmd_penaltydlmd)
        ret = df + np.append(np.zeros(3), dxm_penalty + dlmd_penalty)
        return ret


class DRTSettings:
    def __init__(self):
        self.lmd = 1e-3  # lambda
        self.shape_factor = 0.5  # relative width of radial basis function
        # normally, ratio_out >= ratio_drt >= 1.0
        self.ratio_drt = 1.0  # frequency / time_constant(tau) sampling density ratio of DRT over EIS measurement
        self.ratio_out = 10.0  # frequency / time_constant sampling density ratio of output data over EIS measurement
        self.freq_extension = (0.0, 0.0)
        # freq decades that the drt range is larger than the EIS on low-freq(long-tau) and high-freq(short-tau) side
        self.out_extension = (1.0, 1.0)
        # freq decades that the drt output range is larger than drt freq on
        #     low-freq(long_tau) and high-freq(short-tau) side
        self.solution_rel_err = 0.0  # relative error of the DRT intermediate result as a solver terminating condition
        self.sup_L = [0.0, 0.0]  # suppress predicted inductance if not zero

        self.auto_lmd = True  # Enable increasing lambda for frequencies with lower weights
        # self.hyper_lmd_iter = 0  # hyper_lambda algorithm iterations in drt_tknv and drt_iterative
        self.auto_lmd_radius_dec = 1.0  # smoothing factor for increasing lambda for frequencies with lower weights
        self.iter_count = 1  # drt weighting iterations in drt_iterative
        self.bad_point_suppress = 2  # drt weighting order in drt_iterative
        self.err_thresh = 0.005  # error threshold in drt_iterative
        self.trim_thresh = 0.002  # bad point trimming threshold in drt_iterative
        self.weight_update_ratio = 0.5  # drt weighting update ratio in drt_iterative
        self.false_peak_suppress = True  # suppress peaks at hi-end and lo-end frequencies in drt_iterative
        self.max_trim = 15  # maximum number of EIS frequency points to trim
        self.trim_level = 0.8  # trim terminating condition, higher trim_level could result in more points be trimmed
        self.trim_unskip = 5  # number of points that are exempt from the trim_level condition check


class DRTAnalyzer:
    def __init__(self):
        self.tau_drt = None
        self.tau_out = None
        self.mu = None
        self.settings = DRTSettings()
        self.mf = MatFactory()

    @staticmethod
    def solve_boxcqp(H, b, bounds, epsilon=1e-14, max_iter_unbounded=-1, max_iter_bounded=100, iter_inter=20):
        l_b = len(b)  # length of vector b
        r_H, c_H = len(H), len(H[0])  # number of rows and columns of matrix H
        r_bounds = len(bounds)
        c_bounds = len(bounds[0])
        if not (r_H == c_H == l_b == r_bounds and c_bounds == 2):  # dimension check
            raise IndexError('Dimension mismatch in matrix H, vector b and bounds'
                             '(n rows, 2 columns: lower and upper bounds)!')
        v1 = np.zeros(r_H)  # lambda, associated with the lower bounds
        v2 = np.zeros(r_H)  # mu, associated with the upper bounds
        bounds = np.array(bounds).T
        lower = bounds[0]
        upper = bounds[1]
        x = DRTAnalyzer.solve_pcg(H, b, epsilon=epsilon, max_iter=max_iter_unbounded)

        def assemble_x(x_sub, lower, upper, marker):
            # len(marker) == len(lower) == len(upper) == r_H
            ind_x_sub = 0
            r_H = len(marker)
            x = np.zeros(r_H)
            for ind in range(r_H):
                if marker[ind] == 0:
                    x[ind] = x_sub[ind_x_sub]
                    ind_x_sub += 1
                elif marker[ind] == -1:
                    x[ind] = lower[ind]
                else:
                    x[ind] = upper[ind]
            return x

        def submatrix(A, indices):
            r_A = len(A)
            c_A = len(A[0])
            max_ind = max(indices)
            min_ind = min(indices)
            if r_A == c_A >= max_ind >= min_ind >= 0:
                A = np.array(A)
                return A[np.ix_(indices, indices)]
            else:
                raise IndexError('submatrix: Improper index!')

        for ite in range(max_iter_bounded):
            # update the marker
            marker = np.zeros(r_H)  # if marker[i] == 0, x[i] lies within its bounds;
            #                                         1: x[i] is beyond its upper bound
            #                                        -1: x[i] is beyond its lower bound
            marker -= (x < lower) + ((x == lower) * (v1 >= 0))  # logic operations: + means or, * means and
            marker += (x > upper) + ((x == upper) * (v2 >= 0))  # logic operations: + means or, * means and
            if np.alltrue(marker == 0):  # the solution lies within bounds already
                break
            # calculate x
            ind_marker0 = np.squeeze(np.argwhere(marker == 0))
            if ite > max_iter_bounded - 2:
                iter_inter = max_iter_unbounded
            x_sub = DRTAnalyzer.solve_pcg(submatrix(H, ind_marker0),
                              (b - H @ (lower * (marker < 0) + upper * (marker > 0)))[ind_marker0],
                              epsilon=epsilon, max_iter=iter_inter)
            x = assemble_x(x_sub, lower, upper, marker)
            # calculate v1 and v2 (this does waste some computation...)
            v1 = (H @ x - b) * (marker < 0)
            v2 = (-H @ x + b) * (marker > 0)
            # solution check
            if np.alltrue(((lower <= x) * (x <= upper)) * (marker == 0)
                          + ((v1 >= 0) * (marker < 0)) + ((v2 >= 0) * (marker > 0))):
                break
        x = np.minimum(upper, np.maximum(lower, x))
        return x

    @staticmethod
    def solve_pcg(H, b, epsilon=1e-14, max_iter=-1):
        # Solve using the preconditioned conjugate gradient method
        # find x st. min( 1/2 * xT * H * x - bT * x ) or find x st. H * x = b
        # if positive_result is True, every dimension in the solution vector x must be positive
        if b.T @ b == 0.0:
            return np.zeros(len(b))
        H = np.array(H)
        b = np.array(b)
        l_b = len(b)  # length of vector b
        r_H, c_H = len(H), len(H[0])  # number of rows and columns of matrix H
        if not (r_H == c_H == l_b):  # dimension check
            raise IndexError('Dimension mismatch in matrix H and vector b!')
        e2 = epsilon * epsilon
        x = np.array([0.0] * r_H)  # initialize the result vector
        M = np.eye(r_H) * H  # *: point-wise multiplication, @: matrix multiplication
        m = M @ np.ones(r_H)  # get an array of the diagonal elements of H
        r = b  # initialize the residual vector
        z = r / m
        p = z  # initialize the search direction vector
        if max_iter == -1:
            max_iter = r_H
        for ite in range(max_iter):
            Hp = H @ p
            alpha = (z.T @ r) / (p.T @ Hp)
            if ite % 10 == 1 and (alpha * alpha * p.T @ p) / (b.T @ b) < e2:
                break  # check terminating condition every some iterations
            x = x + alpha * p
            r_new = r - alpha * Hp
            z_new = r_new / m
            beta = (z_new.T @ r_new) / (z.T @ r)
            p = z_new + beta * p
            r = r_new  # update r
            z = z_new  # update z
        # else:
        #     print('In PCG: max iter reached!')
        return x

    def calc_mu(self, shape_factor, freq):
        # freq can be frequencies or time constants
        if shape_factor == np.inf:
            mu = np.inf
        else:
            freq = np.array(freq)
            log10_f_step = np.log(freq[0] / freq[-1]) / np.log(10) / (freq.size - 1)
            mu = shape_factor ** 2 / log10_f_step
        return abs(mu)

    @staticmethod
    def valid_indices(f, zre, zim, radius=2, max_trim=15, trim_level=0.5, trim_unskip=0, is_lnf=False):

        # def linreg(x, y):
        #     # y^ = a + b * x
        #     x = np.array(x)
        #     y = np.array(y)
        #     if len(x) != len(y):
        #         raise ValueError('len(x) is not equal to len(y).')
        #     b = (np.average(x * y) - np.average(x) * np.average(y)) / (np.average(x * x) - np.average(x) ** 2)
        #     a = np.average(y) - b * np.average(x)
        #     return a, b

        def polyreg(x, y, deg=2):
            x = np.array(x)
            y = np.array(y)
            n = len(x)
            if n != len(y):
                raise ValueError("x and y must have equal length, but len(x) = " + str(n) + ", len(y) = " + str(len(y)))
            if deg >= n:
                deg = n - 1
            X = np.ones([deg + 1, n])
            for d in range(1, deg + 1):
                X[d] = X[d - 1] * x
            X = X.T
            coefs = np.linalg.solve(X.T @ X, X.T @ y)
            return coefs  # y^ = sum(coef[i] * x^i, 0 <= i <= deg)

        def get_bias(f, zre, zim, radius=2, skip=None):
            if skip is None:
                skip = [False, ] * len(f)
            if radius < 2:
                radius = 2
            f = np.array(f)  # 1D arrays assumed
            zre = np.array(zre)
            zim = np.array(zim)
            if not len(f) == len(zre) == len(zim):
                raise ValueError("Dimensions of f, zre and zim does not match.")
            if not is_lnf:
                lnf = np.log(f)
            else:
                lnf = np.array(f)
            n = len(f)
            are = np.zeros(n)
            bre = np.zeros(n)
            cre = np.zeros(n)
            aim = np.zeros(n)
            bim = np.zeros(n)
            cim = np.zeros(n)
            for i in range(radius):
                if not skip[i]:
                    # head
                    mask = np.ones(2 * radius + 1, dtype=bool)
                    mask[i] = False  # edge excluded
                    are[i], bre[i], cre[i] = polyreg(lnf[0: 2 * radius + 1][mask], zre[0: 2 * radius + 1][mask], deg=2)
                    aim[i], bim[i], cim[i] = polyreg(lnf[0: 2 * radius + 1][mask], zim[0: 2 * radius + 1][mask], deg=2)
                n_tail = n - 1 - i
                if not skip[n_tail]:
                    # tail
                    mask = np.ones(2 * radius + 1, dtype=bool)
                    mask[2 * radius - i] = False  # edge excluded
                    are[n_tail], bre[n_tail], cre[n_tail] = polyreg(lnf[n - 2 * radius - 1: n][mask],
                                                                    zre[n - 2 * radius - 1: n][mask], deg=2)
                    aim[n_tail], bim[n_tail], cim[n_tail] = polyreg(lnf[n - 2 * radius - 1: n][mask],
                                                                    zim[n - 2 * radius - 1: n][mask], deg=2)
            mask = np.ones(2 * radius + 1, dtype=bool)
            mask[radius] = False  # center excluded
            for i in range(radius, n - radius):
                if skip[i]:
                    continue
                # estimate zre
                are[i], bre[i], cre[i] = polyreg(lnf[i - radius: i + radius + 1][mask],
                                                 zre[i - radius: i + radius + 1][mask], deg=2)
                # estimate zim
                aim[i], bim[i], cim[i] = polyreg(lnf[i - radius: i + radius + 1][mask],
                                                 zim[i - radius: i + radius + 1][mask], deg=2)
            bias = zre - are - lnf * (bre + cre * lnf) + (zim - aim - lnf * (bim + cim * lnf)) * 1j
            # z_smooth = zre + 1j * zim - bias
            # folder = 'E:/OneDrive/Leo/OneDrive/文档/研二/SOFC/成果/期刊会议论文/Papers/EISART/'
            # from util_io import save_eis_data
            # save_eis_data(folder + 'Remaining' + str(len(f)) + 'Pts.txt', f, z_smooth.real, z_smooth.imag, delim='\t')
            # save_eis_data(folder + 'BiasAt' + str(len(f)) + 'Pts.txt', f, bias.real, bias.imag, delim='\t')
            return bias

        if radius < 0:
            raise ValueError("\'radius\' must be an integer no less than 1.")
        n = len(f)
        mask = np.arange(n)
        mask_last = mask * 1
        aver_bias_last = 1e100
        skip = [False, ] * n
        bias = np.ones(n) * 1e100
        for i in range(max_trim):
            bias_new = get_bias(f[mask], zre[mask], zim[mask], radius=radius, skip=skip)
            if bias_new.size < 2:
                break
            bias = np.where(skip, bias, bias_new)
            aver_bias = np.average(np.abs(bias))
            index = np.argmax(np.abs(bias))
            if aver_bias > aver_bias_last * trim_level and i > trim_unskip:
                mask = mask_last * 1
                break
            mask_last = mask * 1
            mask = np.delete(mask, index)  # remove the point with the largest bias
            bias = np.delete(bias, index)  # remove the point with the largest bias
            aver_bias_last = aver_bias
            skip = [True, ] * len(mask)
            for j in range(-radius, radius):
                # example: radius = 2, j in (-2, -1, 0, 1)
                # elements to recalculate: (-2, -1, 1, 2)
                # index change due to the deletion of the 0th element: (-2, -1, 0, 1)
                # so this marks just the right elements to recalculate
                index_recalc = index + j
                if index_recalc < 0 or index_recalc > len(mask) - 1:
                    continue
                skip[index_recalc] = False
        return mask

    @staticmethod
    def kk_transform(f_eis, z_eis_re, z_eis_im, weights=None, tau_ext_dec=0.5, mu_thresh=0.3):
        # based on M. Schönleber, D. Klotz, and E. Ivers-Tiffée, Electrochim. Acta, 131, 20–27 (2014).
        lntau_min = -np.log(2 * np.pi * np.max(f_eis)) - np.log(10) * tau_ext_dec
        lntau_max = -np.log(2 * np.pi * np.min(f_eis)) + np.log(10) * tau_ext_dec
        n = len(f_eis)
        if weights is None or len(weights) != n:
            weights = f_eis * 0 + 1
        A_last = np.ones(1)
        x = np.ones(1)
        for m in range(3, n + 1):
            tau = np.exp(np.linspace(lntau_min, lntau_max, m))
            # fit Z with serial R_inf, L and R//C elements
            # Z = sum(R / (1 + j*omega*tau)), known tau, find R
            # A @ x = Z, x = [R_inf, L_self, L_wire, R (vector)]
            # A.shape == [n, 3 + m]
            omegatau = 2 * np.pi * np.vstack([tau] * n) * np.vstack([f_eis] * m).T
            A = 1. / (1. + 1j * omegatau)
            A = np.vstack([np.ones(n), 2j * np.pi * f_eis * 1e-6, 2 * np.pi * f_eis * 1e-6, A.T]).T
            H = A.real.T @ np.diag(weights * weights) @ A.real + A.imag.T @ np.diag(weights * weights) @ A.imag
            b = z_eis_re @ np.diag(weights * weights) @ A.real + z_eis_im @ np.diag(weights * weights) @ A.imag
            try:
                res = qp(H.real, b)
                x = res[0]  # only for qp (solve_qp) in module quadprog, not qpsolvers
            except ValueError:
                break
            A_last = A
            x_act = x * 1
            x_act[:3] *= 0
            x_act = x_act / np.max(np.abs(x_act))
            x_act = np.maximum(-1, np.minimum(x_act, 1))
            mu = 1 - np.sum(-x_act * (x < 0) + 0.1) / np.sum(x_act * (x >= 0) + 0.1)
            z = z_eis_re + 1j * z_eis_im
            print('m = ', m, 'mu = ', mu, 'err = ', 100 * np.average(np.abs(A @ x - z) / np.abs(z)), '%')
            if mu < mu_thresh:
                break
        z_kk = A_last @ x
        return z_kk.real, z_kk.imag

    def drt_tknv(self, f_eis, z_eis_re, z_eis_im, weights=None,
                 R_inf_set=None, L_set=None):
        # weights are used for the evaluation of matrix H and thus the DRT: how much each eis data point is considered

        if not hasattr(L_set, '__len__'):
            L_set = [L_set, None]
        # if hyper_lmd_iter is None:
        #     hyper_lmd_iter = self.settings.hyper_lmd_iter
        auto_lmd = self.settings.auto_lmd
        auto_lmd_radius_dec = self.settings.auto_lmd_radius_dec
        false_peak_suppress = self.settings.false_peak_suppress

        l_eis = len(f_eis)
        l_drt = int(l_eis * self.settings.ratio_drt + 0.5)
        l_out = int(l_eis * self.settings.ratio_out + 0.5)
        drt_logf_ext_lo = self.settings.freq_extension[0] * np.log(10)
        drt_logf_ext_hi = self.settings.freq_extension[1] * np.log(10)
        out_logf_ext_lo = self.settings.out_extension[0] * np.log(10)
        out_logf_ext_hi = self.settings.out_extension[1] * np.log(10)
        lntaumin = -np.log(np.max(f_eis) * 2 * np.pi)
        lntaumax = -np.log(np.min(f_eis) * 2 * np.pi)
        lntau_drt = np.linspace(lntaumin - drt_logf_ext_hi, lntaumax + drt_logf_ext_lo, l_drt)
        tau_drt = np.exp(lntau_drt)
        tau_out = np.exp(np.linspace(lntaumin - drt_logf_ext_hi - out_logf_ext_hi,
                                     lntaumax + drt_logf_ext_lo + out_logf_ext_lo, l_out))  # ascending order
        self.tau_drt = tau_drt
        self.tau_out = tau_out
        mu = self.calc_mu(self.settings.shape_factor, tau_drt)
        if mu == np.inf or self.settings.lmd == 0:
            inf_mu = True
        else:
            inf_mu = False
            self.mu = self.mf.setmu(mu)  # setmu accepts a target mu and returns tha actually set mu
        Are, Aim, M, phi, matrices = self.mf.getAMP(f_eis, tau_drt, tau_out, inf_mu=inf_mu)
        N, dphi, d2phi, L = matrices[:4]

        lmd_vector0 = self.settings.lmd * np.ones(l_drt)
        lmd_vector = lmd_vector0 * 1.0  # copy
        if auto_lmd and (not inf_mu):
            lnw = np.log(2 * np.pi * f_eis)
            blur_kern = np.exp(-((lnw - np.median(lnw)) / np.log(10.) / auto_lmd_radius_dec) ** 2)
            blur_kern /= np.sum(blur_kern)
            if weights is None:
                weights = np.ones(len(f_eis))
            smoothed_weights = convolve(weights - 1, blur_kern, mode='same') + 1
            if lnw[0] > lnw[-1]:
                lnw = np.flip(lnw)
                smoothed_weights = np.flip(smoothed_weights)
            sw_interp = interp1d(lnw, smoothed_weights, bounds_error=False,
                                 fill_value=(smoothed_weights[0], smoothed_weights[-1]))
            sw = sw_interp(-lntau_drt)
            lmd_vector /= np.maximum(sw ** 16, self.settings.lmd)

            # # the old auto_lmd block: no later than 2021.02
            # # Gavrilyuk AL, Osinkin DA, Bronin DI. On a variation of the Tikhonov regularization method for calculating the distribution function of relaxation times in impedance spectroscopy. Electrochim Acta 2020;354:136683.
            # # https://doi.org/10.1016/j.electacta.2020.136683.
            # # Modification 1: lambda (lmd) is logarithmically mapped before the solver
            # # Modification 2: the input impedance z is normalized to an average magnitude of 1.0 before the solver
            # z_magnification = 0.4  # adjust the relative effect of penalty on xm
            # len_tau_drt = tau_drt.size
            # A = Are + 1j * Aim
            # z = z_eis_re + 1j * z_eis_im
            # if R_inf_set is not None:
            #     z -= R_inf_set
            # if L_set[0] is not None:
            #     z -= 2j * np.pi * f_eis * L_set[0]
            # if L_set[1] is not None:
            #     z -= 2 * np.pi * f_eis * L_set[1]
            # z_aver = np.average(np.abs(z)) / z_magnification
            # z /= z_aver
            # D = np.diag(-0.5 * np.ones(len_tau_drt - 1), 1) + \
            #     np.diag(-0.5 * np.ones(len_tau_drt - 1), -1) + \
            #     np.diag(np.ones(len_tau_drt))
            # dlntau = (lntaumax + drt_logf_ext_lo - lntaumin + drt_logf_ext_hi) / l_drt
            # D = D / dlntau
            # loss = LossAutoLmd(A, f_eis, z, weights, L, D,
            #                    fit_R_inf=R_inf_set is None,
            #                    fit_L_self=L_set[0] is None,
            #                    fit_L_wire=L_set[1] is None)
            # initial_guess = np.zeros(3 + len_tau_drt * 2)
            # if L_set[0] is not None:
            #     initial_guess[0] = L_set[0] / z_aver * 1e6  # L in uH in LossAutoLmd
            # if L_set[1] is not None:
            #     initial_guess[1] = L_set[1] / z_aver * 1e6  # L in uH in LossAutoLmd
            # if R_inf_set is not None:
            #     initial_guess[2] = R_inf_set / z_aver
            #
            # bounds = [[float('-inf'), float('inf')]] * 2 + [[0.0, float('inf')]] + [[0.0, float('inf')]] * l_drt + \
            #          [[float('-inf'), float('inf')]] * l_drt
            # res = minimize(loss.fnc, initial_guess, method='TNC', jac=loss.grad, bounds=bounds,
            #                options={'accuracy': self.settings.solution_rel_err, 'maxiter': 10000})
            # x = res.x
            #
            # # from scipy.optimize import least_squares
            # # bounds = ([0.0] * (3 + l_drt * 2), [float('inf')] * (3 + l_drt * 2))
            # # bounds[0][0] = float('-inf')
            # # bounds[0][1] = float('-inf')
            # # res = least_squares(loss.fnc, initial_guess, jac=loss.grad, bounds=bounds)
            # # x = res[0]
            #
            # L_self, L_wire, R_inf, xm, lmd = x[0] * 1e-6, x[1] * 1e-6, x[2], x[3: 3 + len_tau_drt], \
            #                                  np.exp(x[3 + len_tau_drt:])
            # L_self *= z_aver
            # L_wire *= z_aver
            # R_inf *= z_aver
            # xm *= z_aver
        # else:
        # x = [None]
        # for i in range(hyper_lmd_iter + 1):  # Un-comment to resume the hyper-lambda code
        # Hyper-lambda algorithm (modified)
        # Effat MB, Ciucci F. Electrochimica Acta Bayesian and Hierarchical Bayesian Based Regularization for Deconvolving the Distribution of Relaxation Times from Electrochemical Impedance Spectroscopy Data. Electrochim Acta [Internet]. 2017;247:1117–29. Available from: http://dx.doi.org/10.1016/j.electacta.2017.07.050
        H, b = self.mf.getHb(Are, Aim, M, N * 0.0, f_eis=f_eis, z_eis_re=z_eis_re, z_eis_im=z_eis_im,
                             lmd=lmd_vector, R_inf_set=R_inf_set, L_set=L_set,
                             L_penalty=self.settings.sup_L, weights=weights)

        # my own solver
        # x = self.solve_boxcqp(H, b, bounds=[[0, 1e308], ] * len(b),
        #                       max_iter_unbounded=-1, max_iter_bounded=200, iter_inter=50)

        try:
            # the quadprog solver
            # Literature: from https://pypi.org/project/quadprog/
            # D. Goldfarb and A. Idnani (1983). A numerically stable dual method for solving strictly convex quadratic programs. Mathematical Programming, 27, 1-33.
            bounding_mask = np.eye(len(b))
            bounding_mask[0, 0] = 0
            bounding_mask[1, 1] = 0
            # L_self and L_wire are not subject to the bounding
            res = qp(H, b, bounding_mask, np.zeros(len(b)))
            x = res[0]  # only for qp (solve_qp) in module quadprog, not qpsolvers
            # x = -res  # only for qp (solve_qp) in module qpsolvers, not quadprog
        except ValueError:
            # the scipy minimizer
            magnification = 1e3 / np.average(b)
            loss = Loss(H, b * magnification)
            initial_guess = b
            initial_guess /= np.average(H, axis=1)
            bounds = [[float('-inf'), float('inf')]] * 2 + [[0.0, float('inf')]] + [[0.0, float('inf')]] * l_drt
            if L_set[0] is not None:
                temp = L_set[0] * 1e6
                bounds[0] = [temp, temp]
                initial_guess[0] = temp
            if L_set[1] is not None:
                temp = L_set[1] * 1e6
                bounds[1] = [temp, temp]
                initial_guess[1] = temp
            if R_inf_set is not None:
                bounds[2] = [R_inf_set, R_inf_set]
                initial_guess[2] = R_inf_set
            res = minimize(loss.fHb, initial_guess, method='TNC', jac=loss.grad, bounds=bounds,
                           options={'accuracy': self.settings.solution_rel_err, 'maxiter': 10000})
            x = res.x / magnification

        # the scipy lsq_linear solver (slow!!!)
        # bounds = [0.0, float('inf')]
        # res = lsq_linear(H, b, bounds, max_iter=10000, lsq_solver='exact')
        # x = res.x

        # the cvxopt quadratic programming optimizer
        # options['show_progress'] = False
        # bounding_mask = np.eye(len(b))
        # bounding_mask[0, 0] = 0
        # bounding_mask[1, 1] = 0
        # res = coneqp(matrix(H), -matrix(b), matrix(-bounding_mask), matrix(np.zeros(len(b))))
        # x = np.array(res['x']).reshape([-1, ])

        # # -- update lambda using the Hyper-lambda algorithm starts
        # # first-order derivative + second-order derivative
        # dx2 = (dphi @ x[3:]) ** 2
        # # dx2 /= np.max(dx2)
        # d2x2 = (d2phi @ x[3:]) ** 2
        # # d2x2 /= np.max(d2x2)
        # act = dx2 + d2x2
        # act = act.reshape([-1, ])
        # # import matplotlib.pyplot as plt
        # # plt.plot(act)
        # # plt.show()
        # # act[0] = act[0] * 3/4 + act[1] * 1/4
        # # act[-1] = act[-1] * 3/4 + act[-2] * 1/4
        # # act[1:-1] = (act[1:-1] * 2 + act[:-2] + act[2:]) / 4  # blur with core (0.25, 0.5, 0.25)
        # aver_act = np.average(act) + 1e-20
        # lmd_min = lmd_vector0 * 1e-1
        # lmd_vector = aver_act * (lmd_vector0 - lmd_min) / (act + aver_act) + lmd_min
        # # -- update lambda using the Hyper-lambda algorithm ends

        # L_self: self induction, positive deviations in Z_im; L_wire: wire induction, positive deviations in Z_re
        # The magnetic field excited by current cables can induce a deviation voltage in voltage tap wires
        L_self, L_wire, R_inf, xm = x[0] * 1e-6, x[1] * 1e-6, x[2], x[3:]

        if false_peak_suppress:
            dxm01 = max(0.0, xm[0] + xm[1] - 2 * xm[2]) / 2
            dxm_12 = max(0.0, xm[-1] + xm[-2] - 2 * xm[-3]) / 2
            dxm01 = 2 * dxm01 / (np.sum(xm[:2]) + 1e-100)
            dxm_12 = 2 * dxm_12 / (np.sum(xm[-2:]) + 1e-100)
            if tau_out[0] < tau_out[-1]:
                dR = dxm01 * (1e-100 + xm[0] * np.sum(phi[:, 0]) + xm[1] * np.sum(phi[:, 1])) * \
                         np.log(tau_out[1] / tau_out[0])

            else:
                dR = dxm_12 * (1e-100 + xm[-1] * np.sum(phi[:, -1]) + xm[-2] * np.sum(phi[:, -2])) * \
                         np.log(tau_out[-2] / tau_out[-1])
            R_inf += dR
            L_self -= dR / (2 * np.pi * np.max(f_eis))
            xm[:2] *= 1 - dxm01
            xm[-2:] *= 1 - dxm_12
        gamma = phi @ xm
        gamma = gamma.reshape([-1, ])
        if L_set[0] is not None:
            L_self = L_set[0]
        if L_set[1] is not None:
            L_wire = L_set[1]
        if R_inf_set is not None:
            R_inf = R_inf_set
        return np.array([L_self, L_wire]), R_inf, gamma, tau_out, phi

    @staticmethod
    def eis_from_drt(L, R_inf, gamma, tau, freq):
        n = len(tau)
        eis = np.zeros(len(freq)) * 1j
        dlnt = (np.log(max(tau) / min(tau))) / (n - 1)
        for i, t in enumerate(tau):
            eis += gamma[i] * dlnt / (1 + 2j * np.pi * freq * t)
        if not hasattr(L, '__len__'):
            eis += 2j * np.pi * freq * L + R_inf
        else:
            eis += 2 * np.pi * freq * (1j * L[0] + L[1]) + R_inf
        return eis

    @staticmethod
    def eis_from_drt_mat(tau, freq):
        n = len(tau)
        m = len(freq)
        dlnt = (np.log(max(tau) / min(tau))) / (n - 1)
        mat = np.zeros([m, n]) * 1j
        for i, f in enumerate(freq):
            mat[i] = dlnt / (1 + 2j * np.pi * f * tau)
        return mat

    def drt_iterative(self, f_eis, z_eis_re, z_eis_im, iter_count=None, bad_point_suppress=None, err_thresh=None,
                      trim_thresh=None, weight_update_ratio=None, false_peak_suppress=None,
                      auto_lmd=None, R_inf_set=None, L_set=None, auto_lmd_radius_dec=1.0,
                      eis_significance=None, trim_level=None, max_trim=None, trim_unskip=None):
        if iter_count is None:
            iter_count = self.settings.iter_count
        if bad_point_suppress is None:
            bad_point_suppress = self.settings.bad_point_suppress
        if err_thresh is None:
            err_thresh = self.settings.err_thresh
        if trim_thresh is None:
            trim_thresh = self.settings.trim_thresh
        if weight_update_ratio is None:
            weight_update_ratio = self.settings.weight_update_ratio
        if auto_lmd is None:
            auto_lmd = self.settings.auto_lmd
        # if hyper_lmd_iter is None:
        #     hyper_lmd_iter = self.settings.hyper_lmd_iter
        if false_peak_suppress is None:
            false_peak_suppress = self.settings.false_peak_suppress
        if trim_level is None:
            trim_level = self.settings.trim_level
        if max_trim is None:
            max_trim = self.settings.max_trim
        if trim_unskip is None:
            trim_unskip = self.settings.trim_unskip
        if not hasattr(L_set, '__len__'):
            L_set = [L_set, None]

        eis_data = z_eis_re + 1j * z_eis_im
        mod2 = np.average(eis_data * np.conj(eis_data)).real

        if eis_significance is not None:
            weights = eis_significance ** 8
        else:
            weights = np.ones(len(f_eis))

        kk_drt_settings = deepcopy(self.settings)
        kk_drt_settings.lmd = 0.0
        kk_drt_settings.shape_factor = -1
        kk_drt_an = DRTAnalyzer()
        kk_drt_an.settings = kk_drt_settings

        L, R_inf, gamma, tau, phi = kk_drt_an.drt_tknv(f_eis, z_eis_re, z_eis_im, weights=weights,
                                                       R_inf_set=R_inf_set, L_set=L_set)
        # L, R_inf_ref, gamma, tau, phi = self.drt_tknv(f_eis, z_eis_re, z_eis_im)
        mat = self.eis_from_drt_mat(tau, f_eis)  # mat has dimensions for complete data
        eis_reg = mat @ gamma + R_inf  # eis_reg == EIS data regenerated from DRT results
        eis_reg = eis_reg.reshape([-1, ])
        if hasattr(L, '__len__'):
            z_L = 2 * np.pi * f_eis * (1j * L[0] + L[1])
        else:
            z_L = 2j * np.pi * f_eis * L
        eis_reg += z_L
        eis_err = eis_data - eis_reg
        # w_eis_err = eis_err * 1
        err2 = np.average(eis_err * np.conj(eis_err)).real
        # w_err2 = err2 * 1
        # w_mod2 = mod2 * 1

        if err2 / mod2 > trim_thresh ** 2:  # if the fit is erroneous, trim some data
            valid_indices = self.valid_indices(f_eis, z_eis_re, z_eis_im, radius=2, max_trim=max_trim,
                                               trim_level=trim_level, trim_unskip=trim_unskip)
            weights *= 1e-30
            weights[valid_indices] *= 1e30
        else:  # the error of fit is small enough
            iter_count = 0  # the for loop below would not be executed (so ignore any warnings of unassigned variables)

        # weights used above are passed on to the iterations. Usable but trimmed points may still be resumed
        inv_weight = 1 / (1e-6 + weights)  # weights that learn over the iterations
        for i in range(iter_count):
            weights = 1 / inv_weight
            weights /= np.max(weights)
            L, R_inf, gamma, tau, phi = kk_drt_an.drt_tknv(f_eis, eis_data.real, eis_data.imag, weights=weights,
                                                           R_inf_set=R_inf_set, L_set=L_set)
            mat2 = self.eis_from_drt_mat(tau, f_eis)  # mat2 may have dimensions for incomplete data
            eis_reg = mat2 @ gamma + R_inf  # eis_reg == EIS data regenerated from DRT results
            eis_reg = eis_reg.reshape([-1, ])
            if hasattr(L, '__len__'):
                z_L = 2 * np.pi * f_eis * (1j * L[0] + L[1])
            else:
                z_L = 2j * np.pi * f_eis * L
            eis_reg += z_L
            eis_err = eis_data - eis_reg

            w_eis_err = eis_err * weights / np.average(weights)
            w_err2 = np.average(w_eis_err * np.conj(w_eis_err)).real
            w_mod2 = np.average(weights * eis_data * np.conj(eis_data)).real / np.average(weights)
            if w_err2 / w_mod2 < err_thresh ** 2:
                break
            weights = w_err2 / (w_err2 + np.abs(w_eis_err) ** 2)  # actually inverted weights

            # apply high-pass filter to weights
            # damp_dec = 0.05
            # damp_dec *= np.log(10)

            kernel = [-0.25, 0.5, -0.25]
            d2weights = convolve(weights - 1, kernel, mode='same')
            d2weights = d2weights / np.max(d2weights)
            weights = np.minimum(0.0, np.maximum(d2weights, -1.0 + 1e-6)) + 1

            inv_weight = (1 - weight_update_ratio) * inv_weight + \
                         weight_update_ratio / weights ** bad_point_suppress
        weights = 1 / inv_weight
        weights /= np.max(weights)

        L, R_inf, gamma, tau, phi = self.drt_tknv(f_eis, z_eis_re, z_eis_im, weights=weights,
                                                  R_inf_set=R_inf_set, L_set=L_set)
        return L, R_inf, gamma, tau, phi, weights

    @staticmethod
    def integrate_drt(gamma, tau, tau_from=None, tau_to=None):
        tau_min = np.min(tau)
        tau_max = np.max(tau)
        if tau_from is None:
            tau_from = tau_min
        if tau_to is None:
            tau_to = tau_max
        tau_from, tau_to = sorted([tau_from, tau_to])  # ensure that tau_from <= tau_to
        gamma = np.array(gamma)
        tau = np.array(tau)
        if tau_from < tau_min:
            print("In cumulative_drt: tau_from adjusted to np.min(tau) = ", tau_min)
            tau_from = tau_min
        if tau_to > tau_max:
            print("In cumulative_drt: tau_to adjusted to np.max(tau) = ", tau_max)
            tau_to = tau_max
        index_flags = np.where(tau_from <= tau, True, False) * np.where(tau <= tau_to, True, False)
        tau_integrate = tau[index_flags]
        tau_integrate = np.insert(tau_integrate, 0, tau_from)
        tau_integrate = np.append(tau_integrate, tau_to)
        gamma_integrate = gamma[index_flags]
        gamma_interp = interp1d(tau, gamma)
        gamma_integrate = np.insert(gamma_integrate, 0, gamma_interp(tau_from))
        gamma_integrate = np.append(gamma_integrate, gamma_interp(tau_to))
        lntau_integrate = np.log(tau_integrate)
        integral = 0.5 * (gamma_integrate[:-1] + gamma_integrate[1:]) * (lntau_integrate[1:] - lntau_integrate[:-1])
        integral = np.sum(integral)
        return integral

    @staticmethod
    def cumulative_drt(gamma, tau):
        gamma = np.array(gamma)
        lntau = np.log(tau)
        dR = 0.5 * (gamma[:-1] + gamma[1:]) * (lntau[1:] - lntau[:-1])
        cR = np.cumsum(dR)  # cumulative R
        cR = np.insert(cR, 0, 0.0)
        if cR[-1] < 0:
            cR -= cR[-1]
        return cR


def create_path(newpath):
    if path.exists(newpath):
        # shutil.rmtree(newpath)
        # print('Specified path already exists!', path)
        return 1
    else:
        makedirs(newpath)
        # print('Specified path created!', path)
        return 0


def single_file(f_in, f_out, root_dir, subfolder, settings):  # depricated since Jan 2020
    path_in = path.join(root_dir, f_in)
    path_out = path.join(root_dir, subfolder, f_out)
    create_path(path.join(root_dir, subfolder))
    eis = np.loadtxt(path_in, delimiter=",", skiprows=0)
    f_eis, z_eis_re, z_eis_im = eis[:, 0], eis[:, 1], eis[:, 2]
    drt_analyzer = DRTAnalyzer()
    drt_analyzer.settings = settings
    L, R_inf, gamma, tau, _, _ = drt_analyzer.drt_iterative(f_eis, z_eis_re, z_eis_im)
    from util_io import save_drt_data
    save_drt_data(path_out, L, R_inf, gamma, tau)
    return L, R_inf, gamma, tau


def main_compare():
    # for testing the util module
    print('start\n')
    import util_io
    import util_plot

    eis_filename = input('Please specify the impedance spectrum file:\n').strip('&').strip(' ').strip('\'').strip('\"')
    eis_data = util_io.load_eis_data(eis_filename).T
    eis_data[1:, :] *= 1

    # f = np.power(10.0, np.linspace(5, -1, 73))
    # omega = 2 * np.pi * f
    # z = 0.3 + 0.125 / (1 + np.power(1j * omega * 2.5e-5, 0.88)) + 0.355 / (1 + np.power(1j * omega * 2.4e-4, 0.82)) \
    #     + 0.157 / (1 + np.power(1j * omega * 1.5e-3, 0.81)) + 0.283 / (1 + np.power(1j * omega * 2.8e-2, 0.91)) \
    #     + 2j * np.pi * f * 0.72e-6 - 2 * np.pi * f * 0e-6 + 2 * np.pi * f ** 2 * 0e-11
    # z *= 1 + (np.random.standard_normal(f.size) + 1j * np.random.standard_normal(f.size)) * 1e-2
    # eis_data = np.vstack([f, z.real, z.imag])

    drt_analyzer = DRTAnalyzer()
    # drt_analyzer.settings.shape_factor = np.inf  # kk_test mode
    # drt_analyzer.settings.lmd = 1e-6
    drt_analyzer.settings.auto_lmd = False
    L1, R_inf1, gamma1, tau1, _ = drt_analyzer.drt_tknv(eis_data[0], eis_data[1], eis_data[2])
    # L1, R_inf1, gamma1, tau1, _, _ = drt_analyzer.drt_iterative(eis_data[0], eis_data[1], eis_data[2], iter_count=1,
    #                                                             bad_point_suppress=4, hyper_lmd_iter=0, auto_lmd=True,
    #                                                             R_inf_set=None, L_set=[None, None])
    drt_analyzer.settings.auto_lmd = True
    L2, R_inf2, gamma2, tau2, phi, w = drt_analyzer.drt_iterative(eis_data[0], eis_data[1], eis_data[2], iter_count=1,
                                                                bad_point_suppress=2, auto_lmd=True,
                                                                R_inf_set=None, L_set=[None, None])
    eis_z1 = drt_analyzer.eis_from_drt(L1, R_inf1, gamma1, tau1, eis_data[0])
    eis_z2 = drt_analyzer.eis_from_drt(L2, R_inf2, gamma2, tau2, eis_data[0])
    from util_ecm import ECM
    ecm = ECM()
    # ecm.n_rq = 5
    # ecm.R_inf_ref = 0.1
    # ecm.L_ref = [0.1e-6, 0.0]
    # ecm.rq_pars['R'] = np.ones(5) * 0.1
    # ecm.rq_pars['tau'] = 1 / (2 * np.pi * 10 ** np.arange(5))
    # ecm.rq_pars['alpha'] = np.ones(5) * 0.9
    ecm_filename = input('Please specify the equivalent circuit file:\n').strip('&').strip(' ').strip('\'').strip('\"')
    ecm.load(ecm_filename)
    gamma3, tau3 = ecm.drt_gen(tau2, drt_analyzer.tau_drt, drt_analyzer.mu, phi)

    eis_format_ids = ((1, 0, 0), (2, 8, 1), (2, 10, 0))
    eis_labels = ('Input', 'Unweighed', 'Weighed')
    drt_format_ids = ((2, 8, 1), (2, 10, 0), (2, 0, 0))
    drt_labels = ('Unweighed', 'Weighed', 'Truth')

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(8, 5), constrained_layout=True)
    util_plot._plot_eis([axs[0][0], axs[1][0]], freq=[eis_data[0]] * 3,
                        z_real=[eis_data[1], eis_z1.real, eis_z2.real],
                        z_imag=[eis_data[2], eis_z1.imag, eis_z2.imag],
                        title='EIS', format_ids=eis_format_ids, 
                        labels=eis_labels,
                         weights=w)
    
    util_plot._plot_drt(axs[0][1], gamma=[gamma1, gamma2, gamma3], tau=[tau1, tau2, tau3],
                        show_tau=True, show_LR=False, 
                        #  labels=drt_labels,
                        title='DRT', format_ids=drt_format_ids)
    axs[0][1].set_ylabel(r'$\gamma\/ / \/ \mathrm{\Omega \bullet cm^2 \bullet ln(s)^{-1}}$')
    
    cdrt1 = drt_analyzer.cumulative_drt(gamma1, tau1) + R_inf1
    cdrt2 = drt_analyzer.cumulative_drt(gamma2, tau2) + R_inf2
    cdrt3 = drt_analyzer.cumulative_drt(gamma3, tau3) + ecm.R_inf_ref
    util_plot._plot_xy(axs[1][1], x=[tau1, tau2, tau3], y=[cdrt1, cdrt2, cdrt3], xlog=True, 
                       format_ids=drt_format_ids,
                       xlabel=r'$\tau$'+' / s', 
                       ylabel='Cumulative '+r'$\gamma \/ / \/ \mathrm{\Omega \bullet cm^2}$', 
                       labels=drt_labels)
    plt.show()

    print('done\n')
    # ref [1]:
    #   T.H. Wan, M. Saccoccio, C. Chen, F. Ciucci, Influence of the Discretization Methods on
    #   the Distribution of Relaxation Times Deconvolution: Implementing Radial Basis Functions with DRTtools,
    #   Electrochimica Acta, 184 (2015) 483-499.


def main_just_plot():
    print('start\n')
    import util_io
    import util_plot

    # eis_filename = 'example_eis.ism'
    # eis_data = util_io.load_eis_data(eis_filename).T
    # eis_data[1:, :] *= 80
    r_ohm = 0.6
    # r_act_an = 0.42
    j = 0.27
    j_ex_an = 0.12
    j_ex_ca = 0.12
    T = 1073
    r_act_an = 8.314 * T / (2 * 96485 * j_ex_an * np.sqrt(1 + (0.5 * j / j_ex_an) ** 2))
    # r_act_ca = 0.157
    r_act_ca = 8.314 * T / (2 * 96485 * j_ex_ca * np.sqrt(1 + (0.5 * j / j_ex_ca) ** 2))
    # r_diff_an = 0.283
    x_H2_TPB = 0.857
    j_limD_f = 2.
    r_diff_an = 8.314 * T / (2 * 96485 * j_limD_f * x_H2_TPB * (1 - x_H2_TPB))

    f = np.power(10.0, np.linspace(5, -1, 73))
    omega = 2 * np.pi * f
    z = r_ohm + 0.26 * r_act_an / (1 + np.power(1j * omega * 2.5e-5, 0.88)) \
        + 0.74 * r_act_an / (1 + np.power(1j * omega * 2.4e-4, 0.82)) \
        + r_act_ca / (1 + np.power(1j * omega * 1.5e-3, 0.81)) \
        + r_diff_an / (1 + np.power(1j * omega * 2.8e-2, 0.91)) \
        + 2j * np.pi * f * 0.36e-60 - 2 * np.pi * f * 0e-6 + 2 * np.pi * f ** 2 * 0e-11
    # z *= 1 + (np.random.standard_normal(f.size) + 1j * np.random.standard_normal(f.size)) * 1e-2
    eis_data = np.vstack([f, z.real, z.imag])
    eis_format_ids = ((1, 0, 0),)
    hdl = util_plot.plot_eis(eis_data[0], eis_data[1], eis_data[2], format_ids=eis_format_ids)
    util_plot.show_plot(hdl)

def main_test_kk():
    from util_io import load_eis_data
    eis_file = input('Please specify the EIS file:\n')
    eis_data = load_eis_data(eis_file)
    drt_an = DRTAnalyzer()
    f = eis_data[:, 0]
    zre = eis_data[:, 1]
    zim = eis_data[:, 2]
    valid_indices = drt_an.valid_indices(f, zre, zim, max_trim=5, trim_level=0.8, trim_unskip=2)
    w = f * 0 + 1e-6
    w[valid_indices] *= 1e6
    zkkre, zkkim = drt_an.kk_transform(f, zre, zim, weights=w)
    from util_plot import plot_eis, show_plot
    hdl = plot_eis([f, f], [zre, zkkre], [zim, zkkim], format_ids=((1, 0, 0), (1, 1, 1)))
    show_plot(hdl)

if __name__ == '__main__':
    main_compare()
