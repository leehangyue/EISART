"""
Equivalent Circuit Model
support Electrochemical Impedance Spectrum generation & fitting
support Distribution of Relaxation Time generation & fitting
Code by Hangyue Li, Tsinghua SOFC Lab
"""

from scipy.signal import find_peaks, convolve
from scipy.optimize import fsolve, minimize, Bounds
from scipy.interpolate import interp1d
import numpy as np

from os.path import dirname, splitext
cd = dirname(__file__)
from sys import path
path.append(cd)
import util_tikhonov
path.pop()
del path

message_via_gui_switch = False


def print_to_gui(string):
    print(string)
    if message_via_gui_switch:
        try:
            import EISART_support
            EISART_support.print_to_gui(string)
        except ImportError:
            pass
        except AttributeError:
            pass


def gerischer_pos_to_id(gerischer_positions):
    gerischer_ids = []
    for i, g_id in enumerate(gerischer_positions):
        g_id = -g_id  # flip
        if g_id == 0:
            pass
        elif g_id > 0:
            gerischer_ids.append(g_id - 1)
        else:
            gerischer_ids.append(g_id)
    return gerischer_ids

class ManualECMPars:  # Manually set bounds/limits of ECM parameters (tau and alpha in version 20201221)
    def __init__(self, n=5):
        self.n = n
        self._pars = []
        for i in range(self.n):
            elem = ManualECMElement()
            self._pars.append(elem)

    def __len__(self):
        return self.n

    def from_str(self, s):
        dict_ = eval(s)  # evaluate the string. The result is expected to be a dictionary
        for key, val in dict_.items():
            setattr(self, key, val)
        _pars_temp = [0, ] * len(self._pars)
        for i, par in enumerate(self._pars):
            m_par = ManualECMElement()
            _pars_temp[i] = m_par.from_str(str(par))
        self._pars = _pars_temp
        return self

    def __str__(self):
        str_ = '{\'n\': ' + str(self.n) + ', \'_pars\': ['
        for par in self._pars:
            str_ += str(par) + ', '
        str_ += '], }'
        return str_

    def load(self, filename):
        from os import path
        if not path.exists(filename):
            # raise ValueError("Plot settings file \'" + filename + '\' does not exist.')
            self.save(filename)
        else:
            with open(filename, 'r') as f:
                s = f.read()
                s = s.replace(',\n\'', ', \'')
                self.from_str(s)

    def save(self, filename):
        with open(filename, 'w+') as f:
            s = str(self)
            s = s.replace(', \'', ',\n\'')
            f.write(s)  # TODO may overwrite without warning

    def _check_index(self, index):  # MATLAB format, index starts from 1
        index = int(index)
        if index < 1 or index > self.n:
            raise ValueError("Index out of range: 1 <= ", index, " <= ", self.n, " must be true.\n"
                             "MATLAB format, index starts from 1")
        return index

    def get(self, index):
        index = self._check_index(index)
        string = str(self._pars[index - 1])
        elem = ManualECMElement()
        return elem.from_str(string)  # deep copy to avoid direct manipulation of self._pars

    def set(self, index, elem):
        index = self._check_index(index)
        if not isinstance(elem, ManualECMElement):
            raise ValueError("elem must be an instance of ManualECMElement.")
        string = str(elem)
        elem = ManualECMElement()
        self._pars[index - 1] = elem.from_str(string)  # deep copy to avoid direct manipulation of self._pars

    def insert(self, index, elem):
        self.n += 1
        index = self._check_index(index)
        if not isinstance(elem, ManualECMElement):
            raise ValueError("elem must be an instance of ManualECMElement.")
        string = str(elem)
        elem = ManualECMElement()
        self._pars.insert(index - 1, elem.from_str(string))  # deep copy to avoid direct manipulation of self._pars

    def append(self, elem):
        self.n += 1
        if not isinstance(elem, ManualECMElement):
            raise ValueError("elem must be an instance of ManualECMElement.")
        string = str(elem)
        elem = ManualECMElement()
        self._pars.append(elem.from_str(string))  # deep copy to avoid direct manipulation of self._pars

    def remove(self, index):
        index = self._check_index(index)
        del self._pars[index - 1]
        self.n -= 1

    def sort_tau(self, reverse=False):
        self._pars.sort(reverse=reverse, key=self._get_tau)

    @staticmethod
    def _get_tau(elem):
        return np.sqrt(elem.taumin * elem.taumax)

    def get_unfixed_indices(self):
        indices = []
        for i in range(self.n):
            if not self._pars[i].fixed:
                indices.append(i)
        indices = np.array(indices) + 1
        return indices

    def get_gerischer_indices(self):
        indices = []
        for i in range(self.n):
            if self._pars[i].isGerischer:
                indices.append(i)
        indices = np.array(indices) + 1
        return indices

    def get_tau_bounds(self):
        bounds = []
        for i in range(self.n):
            bounds.append([self._pars[i].taumin, self._pars[i].taumax])
        bounds = np.array(bounds).T
        return bounds

    def get_alpha_bounds(self):
        bounds = []
        for i in range(self.n):
            bounds.append([self._pars[i].alphamin, self._pars[i].alphamax])
        bounds = np.array(bounds).T
        return bounds


class ManualECMElement:
    def __init__(self):
        self.label = 'unnamed'
        # self._R = 1.0  # Ohm * cm^2
        # self._Rmin = 0.0
        # self._Rmax = 1e15
        # self._tau = 1e-3  # s
        self._taumin = 1e-15
        self._taumax = 1e15
        # self._alpha = 0.9
        self._alphamin = 0.4
        self._alphamax = 1.0
        self.isGerischer = False
        self.fixed = False

    def from_str(self, s):
        dict_ = eval(s)  # evaluate the string. The result is expected to be a dictionary
        for key, val in dict_.items():
            setattr(self, key, val)
        return self

    def __str__(self):
        # convert this instance to a dictionary, then to a string, then remove the '_' for protected items,
        str_ = str(self.__dict__)
        while '_' in str_:
            str_ = str_.replace('_', '')
        return str_

    @staticmethod
    def _check_val(_min, _val, _max):
        # if not _min <= _val <= _max:
        #     raise ValueError("The set value is out of range.")
        return max(_min, min(_val, _max))

    # @property
    # def R(self):
    #     return self._R
    #
    # @R.setter
    # def R(self, R):
    #     self._R = self._check_val(self._Rmin, R, self._Rmax)
    #
    # @property
    # def Rmin(self):
    #     return self._Rmin
    #
    # @Rmin.setter
    # def Rmin(self, Rmin):
    #     if Rmin > self._Rmax:
    #         self._Rmax = Rmin
    #     if Rmin > self._R:
    #         self._R = Rmin
    #     self._Rmin = Rmin
    #
    # @property
    # def Rmax(self):
    #     return self._Rmax
    #
    # @Rmax.setter
    # def Rmax(self, Rmax):
    #     if Rmax < self._Rmin:
    #         self._Rmin = Rmax
    #     if Rmax < self._R:
    #         self._R = Rmax
    #     self._Rmax = Rmax

    # @property
    # def tau(self):
    #     return self._tau
    #
    # @tau.setter
    # def tau(self, tau):
    #     self._tau = self._check_val(self._taumin, tau, self._taumax)

    @property
    def taumin(self):
        return self._taumin

    @taumin.setter
    def taumin(self, taumin):
        if taumin < 1e-150:
            taumin = 1e-150
        if taumin > self._taumax:
            self._taumax = taumin
        # if taumin > self._tau:
        #     self._tau = taumin
        self._taumin = taumin

    @property
    def taumax(self):
        return self._taumax

    @taumax.setter
    def taumax(self, taumax):
        if taumax > 1e150:
            taumax = 1e150
        if taumax < self._taumin:
            self._taumin = taumax
        # if taumax < self._tau:
        #     self._tau = taumax
        self._taumax = taumax

    # @property
    # def alpha(self):
    #     return self._alpha
    #
    # @alpha.setter
    # def alpha(self, alpha):
    #     self._alpha = self._check_val(self._alphamin, alpha, self._alphamax)

    @property
    def alphamin(self):
        return self._alphamin

    @alphamin.setter
    def alphamin(self, alphamin):
        if alphamin < 0.0:
            alphamin = 0.0
        if alphamin > self._alphamax:
            self._alphamax = alphamin
        # if alphamin > self._alpha:
        #     self._alpha = alphamin
        self._alphamin = alphamin

    @property
    def alphamax(self):
        return self._alphamax

    @alphamax.setter
    def alphamax(self, alphamax):
        if alphamax > 1.0:
            alphamax = 1.0
        if alphamax < self._alphamin:
            self._alphamin = alphamax
        # if alphamax < self._alpha:
        #     self._alpha = alphamax
        self._alphamax = alphamax


def apply_bounds(variable, bounds):
    ret = np.maximum(bounds.lb, np.minimum(variable, bounds.ub))
    if not hasattr(ret, '__len__'):
        ret = np.array([ret, ])
    return ret


def indexed_bounds(bounds, indices):
    lb, ub = bounds.lb, bounds.ub
    if hasattr(lb, '__len__') and len(lb) > 1:
        lb, ub = np.array(lb), np.array(ub)
        ret = Bounds(lb[indices], ub[indices])
    else:
        ret = Bounds([lb, ] * len(indices), [ub, ] * len(indices))
    return ret


def merge_bounds(bounds1, bounds2):
    lb1, ub1 = np.array(bounds1.lb), np.array(bounds1.ub)
    lb2, ub2 = np.array(bounds2.lb), np.array(bounds2.ub)
    lb, ub = np.append(lb1, lb2), np.append(ub1, ub2)
    bounds = Bounds(lb, ub)
    return bounds


def limit_bounds_and_unfixed_indices(bounds, unfixed_indices, tau_min=0.0, tau_max=1e15):
    taumin = tau_min * (1 - 1e-10)
    taumax = tau_max * (1 + 1e-10)
    bds = np.array([bounds.lb, bounds.ub]).T
    bds_limited = []
    limited_unfixed_indices = []
    if bds.size == 0:
        return np.array([]), np.array([])
    for i, bd in enumerate(bds):
        if bd[0] > taumax or bd[1] < taumin:
            bds_limited.append([bd[0], bd[1]])
            # bd is completely out of the range [taumin, taumax]
        else:
            bds_limited.append([max(taumin, bd[0]), min(taumax, bd[1])])
            if i in unfixed_indices:
                limited_unfixed_indices.append(i)
    bds_limited = np.array(bds_limited).T
    lim_bounds = Bounds(bds_limited[0], bds_limited[1])
    limited_unfixed_indices = np.array(limited_unfixed_indices)
    return lim_bounds, limited_unfixed_indices


def find_nearest_indices(arr, vals):
    # find the indices of elements in arr that are nearest to each entry in vals
    arr, vals = np.array(arr), np.array(vals)  # raise exceptions if this conversion to np array fails
    if arr.size < 2:
        raise ValueError("arr must be an array with at least 2 elements.")
    if len(arr.shape) != 1:
        raise ValueError("arr must be a 1d array / vector.")
    if vals.size == 1:
        return np.argmin(arr - vals[0])
    elif vals.size > 1:
        ret = vals * 0  # copy size
        for index, val in np.ndenumerate(vals):
            ret[index] = np.argmin(np.abs(arr - val))
    else:
        raise ValueError("vals must contain at least one element")
    return np.array(ret, dtype=int)


class ECM:
    def __init__(self, n_rq=5):
        self.n_rq = n_rq
        self.rq_pars = {'R': np.ones(self.n_rq) * 1e-1,  # Ohm
                        'tau': np.ones(self.n_rq) * 1e-3,
                        'alpha': np.ones(self.n_rq) * 0.95}
        self.alpha_ref = None
        self.R_inf_ref = 0.0  # Ohm
        self._L_ref = 0.0  #
        self.cdrt_ref = None
        self._cdrt_gen = None
        self.gamma_ref = None
        self._gamma_gen = None
        self._gamma_gen_indiv = None
        self.tau_out = None
        self.indices_tau_drt_out = None
        self.phi = None
        self.eis_weights = None
        self.gerischer_ids = []
        self.peak_damp_dec = 0.1
        self.drt_settings = util_tikhonov.DRTSettings()
        self.drt_analyzer = None

    @property
    def cdrt_gen(self):
        if self._cdrt_gen is None:
            cdrt_gen = self.drt_analyzer.cumulative_drt(self.gamma_gen, self.tau_out)
            self._cdrt_gen = cdrt_gen
        return self._cdrt_gen

    @property
    def gamma_gen(self):
        if self._gamma_gen is None:
            gamma_gen, _ = self.drt_gen(self.tau_out, self.drt_analyzer.tau_drt, self.drt_analyzer.mu, self.phi)
            self._gamma_gen = gamma_gen
        return self._gamma_gen

    @property
    def gamma_gen_indiv(self):
        if self._gamma_gen_indiv is None:
            self._gamma_gen_indiv, _ = self.drt_gen(self.tau_out, self.drt_analyzer.tau_drt, 
                                                    self.drt_analyzer.mu, self.phi, indiv_elements=True)
        return self._gamma_gen_indiv

    def __str__(self) -> str:
        return self.save('')

    def save(self, filename):
        ecm = self  # code moved from util_io on 20220105
        sorted_indices = np.argsort(ecm.rq_pars['tau'])
        Rs = ecm.rq_pars['R'][sorted_indices]  # ohm * cm^2
        taus = ecm.rq_pars['tau'][sorted_indices]
        alphas = ecm.rq_pars['alpha'][sorted_indices]
        Qs = np.power(taus, alphas) / Rs
        unsort_indices = np.argsort(sorted_indices)
        gerischer_ids = ecm.gerischer_ids
        if gerischer_ids is not None and gerischer_ids:  # if not empty
            gerischer_ids = unsort_indices[ecm.gerischer_ids]

        if not (filename is None or filename == ''):
            ext = splitext(filename)[1][1:]  # remove the point: '.txt' -> 'txt
            if ext == 'csv':
                delim = ','
            else:
                delim = '\t'
        else:
            delim = '\t'
        header = 'L_self (uH)' + delim + '{:.8e}'.format(ecm.L_ref[0] * 1e6) \
                + '\nL_wire (equiv. uH)' + delim + '{:.8e}'.format(ecm.L_ref[1] * 1e6) \
                + '\nR_inf_ref (Ohm * cm^2)' + delim + '{:.8e}'.format(ecm.R_inf_ref) + '\n'
        header += 'ID' + delim + 'R(Ohm*cm^2)' + delim + 'tau(s)     ' + delim + 'alpha      ' + delim \
                + 'Q(s^alpha*Ohm^-1*cm^-2)'
        data_to_save = np.vstack([Rs, taus, alphas, Qs]).T

        string = header + '\n'
        for i in range(len(sorted_indices)):
            string += str(i + 1)  # so that the element IDs begin with 1 not 0
            if i in gerischer_ids:
                string += 'G'
                data_to_save[i][2:] = 0.0
            string += delim
            for dat in data_to_save[i]:
                string += '{:<10.5g}'.format(dat) + delim
            string = string[:-1] + '\n'
        return string

    def load(self, filename):
        if not (filename is None or filename == ''):
            ext = splitext(filename)[1][1:]  # remove the point: '.txt' -> 'txt
            if ext == 'csv':
                delim = ','
            else:
                delim = '\t'
        else:
            raise ValueError('Filename not specified!')
        with open(filename, 'r') as f:
            lines = f.readlines()
        self._L_ref = [0.0, 0.0]
        ecm_elem_list = []  # [ecm_element1, ecm_element2, ...]
        for line in lines:
            line.replace('\r\n', '\n')
            if not line.strip('\n').replace(' ', '').replace(delim, ''):
                continue
            if 'L_self (uH)' in line:
                self._L_ref[0] = float(line.strip('\n').split(delim)[1].strip(' ')) * 1e-6
                continue
            if 'L_wire (equiv. uH)' in line:
                self._L_ref[1] = float(line.strip('\n').split(delim)[1].strip(' ')) * 1e-6
                continue
            if 'R_inf_ref (Ohm * cm^2)' in line:
                self.R_inf_ref = float(line.strip('\n').split(delim)[1].strip(' '))
                continue
            if 'ID' in line:
                continue
            entries = line.strip('\n').split(delim)
            entries = [entry.strip(' ') for entry in entries]
            # entries = [ID, R, tau, alpha, Q]
            # ecm_elem = [isGerischer, R, tau, alpha]
            ecm_elem = [False, float(entries[1]), float(entries[2]), float(entries[3])]
            if 'G' in entries[0]:
                ecm_elem[0] = True
            ecm_elem_list.append(ecm_elem)
        self.n_rq = len(ecm_elem_list)
        ecm_elem_arr = np.array(ecm_elem_list)
        self.gerischer_ids = np.argwhere(ecm_elem_arr[:, 0] > 0.5).T[0].tolist()
        self.rq_pars['R'] = ecm_elem_arr[:, 1]
        self.rq_pars['tau'] = ecm_elem_arr[:, 2]
        self.rq_pars['alpha'] = ecm_elem_arr[:, 3]

    @property
    def L_ref(self):
        return self._L_ref

    @L_ref.setter
    def L_ref(self, L_ref_set):
        if not hasattr(L_ref_set, '__len__'):
            self._L_ref = [L_ref_set, 0.0]
        else:
            if len(L_ref_set) >= 2:
                self._L_ref = L_ref_set[:2]
            elif len(L_ref_set) == 1:
                self._L_ref = [L_ref_set[0], 0.0]
            else:
                raise ValueError("")

    def fit_eis(self, freq, zre, zim, silent=False, refine_alpha=False, gerischer_ids=None, inherit=False,
                fix_tau=False, eis_significance=None, fit_elementwise=False, m_ecm_pars=None,
                R_inf_set=None, L_set=None):
        if self.cdrt_ref is None or self._cdrt_gen is None or self.gamma_ref is None or \
                self._gamma_gen is None or self.tau_out is None or self.eis_weights is None:
            inherit = False

        if eis_significance is None:
            eis_significance = np.ones(len(freq))

        if not silent:
            print_to_gui('Evaluating DRT...')
        drt_analyzer = util_tikhonov.DRTAnalyzer()
        drt_analyzer.settings = self.drt_settings
        self.drt_analyzer = drt_analyzer
        L, R_inf, gamma_ref, tau_out, phi, weights = drt_analyzer.drt_iterative(freq, zre, zim,
                                                                                eis_significance=eis_significance,
                                                                                R_inf_set=R_inf_set, L_set=L_set)
        self._L_ref = L
        self.R_inf_ref = R_inf
        self.gamma_ref = gamma_ref
        self.tau_out = tau_out
        self.phi = phi
        self.eis_weights = weights
        tau_drt = drt_analyzer.tau_drt
        self.indices_tau_drt_out = find_nearest_indices(tau_out, tau_drt)
        tau_min = np.min(tau_drt) * (1 + 1e-10)
        tau_max = np.max(tau_drt) * (1 - 1e-10)
        mu = drt_analyzer.mu

        if not silent:
            print_to_gui('Estimating tau and R...')
        cdrt_ref = drt_analyzer.cumulative_drt(gamma_ref, tau_out)
        self.cdrt_ref = cdrt_ref
        if isinstance(m_ecm_pars, ManualECMPars):
            m_ecm_pars_ = ManualECMPars()
            m_ecm_pars_.from_str(str(m_ecm_pars))
            m_ecm_pars_.sort_tau(reverse=True)
            gerischer_ids = gerischer_pos_to_id(m_ecm_pars_.get_gerischer_indices() * -1)
            tau_bounds = m_ecm_pars_.get_tau_bounds()
            tau_bounds = Bounds(tau_bounds[0], tau_bounds[1])
            unfixed_indices = m_ecm_pars_.get_unfixed_indices() - 1
            tau_bounds, unfixed_indices = \
                limit_bounds_and_unfixed_indices(tau_bounds, unfixed_indices,
                                                 tau_min=np.min(tau_drt), tau_max=np.max(tau_drt))
            alpha_bounds = m_ecm_pars_.get_alpha_bounds()
            alpha_bounds = Bounds(alpha_bounds[0], alpha_bounds[1])
            self.n_rq = m_ecm_pars_.n
            taus = np.sqrt(tau_bounds.lb * tau_bounds.ub)
            peak_indices = find_nearest_indices(tau_out, taus)
            self.rq_pars['tau'] = taus * 1
            Rs, _ = self.estimate_Rs(gamma_ref, cdrt_ref, peak_indices)
            self.rq_pars['R'] = Rs
            if not inherit:
                alpha = self.estimate_alphas(gamma_ref, peak_indices, Rs)
                self.alpha_ref = apply_bounds(alpha, alpha_bounds)
                self.rq_pars['alpha'] = self.alpha_ref
        else:
            m_ecm_pars_ = None
            peak_indices = self.find_gamma_peaks(gamma_ref, tau_out, max_n_peaks=self.n_rq)
            self.n_rq = peak_indices.size
            unfixed_indices = np.arange(self.n_rq)
            tau_bounds = Bounds([tau_min, ] * self.n_rq, [tau_max, ] * self.n_rq)
            tau_bounds, unfixed_indices = \
                limit_bounds_and_unfixed_indices(tau_bounds, unfixed_indices,
                                                 tau_min=np.min(tau_drt), tau_max=np.max(tau_drt))
            alpha_bounds = Bounds([0.4, ] * self.n_rq, [1.0, ] * self.n_rq)
            Rs, _ = self.estimate_Rs(gamma_ref, cdrt_ref, peak_indices)
            self.rq_pars['R'] = Rs
            self.rq_pars['tau'] = apply_bounds(tau_out[peak_indices], tau_bounds)
            if not inherit:
                alpha = self.estimate_alphas(gamma_ref, peak_indices, Rs)
                self.alpha_ref = np.maximum(0.4, np.minimum(alpha, 1.0))
                self.rq_pars['alpha'] = self.alpha_ref
        R_bounds = Bounds([0, ] * self.n_rq, [float('inf'), ] * self.n_rq)  # not fully effective
        # R_bounds_unfixed = *** Not applicable ***
        if len(unfixed_indices) == 0:  # all parameters are fixed, then simulate according to the specified parameters
            self.cdrt_ref = drt_analyzer.cumulative_drt(gamma_ref, tau_out)
            eis_gen = self.eis_gen(freq)
            _, _, gamma_gen, _, _, _ = drt_analyzer.drt_iterative(freq, eis_gen.real, eis_gen.imag,
                                                                  R_inf_set=R_inf_set, L_set=L_set)
            self._cdrt_gen = drt_analyzer.cumulative_drt(gamma_gen, tau_out)
            return
        tau_bounds_unfixed = indexed_bounds(tau_bounds, unfixed_indices)
        alpha_bounds_unfixed = indexed_bounds(alpha_bounds, unfixed_indices)
        if gerischer_ids is not None and gerischer_ids:  # is not empty
            self.gerischer_ids = np.array(gerischer_ids) % self.n_rq
        else:
            self.gerischer_ids = []
        total_R = self.cdrt_ref[-1] - self.cdrt_ref[0]
        n = self.n_rq

        z_ref = zre + 1j * zim
        if inherit:
            self.rq_pars['alpha'] = self.alpha_ref
        else:
            if refine_alpha:  # deprecated
                if not silent:
                    print_to_gui('Refining alpha...')
                tau_out_short = tau_out[self.indices_tau_drt_out]

                def _loss_alpha(alphas):
                    alphas = apply_bounds(alphas, alpha_bounds_unfixed)
                    self.rq_pars['alpha'][unfixed_indices] = alphas
                    gamma_gen, _ = self.drt_gen(tau_out_short, tau_drt, mu, phi)
                    # self._gamma_gen = gamma_gen
                    cdrt_gen = drt_analyzer.cumulative_drt(gamma_gen, tau_out_short)
                    # self._cdrt_gen = cdrt_gen
                    dev = cdrt_gen - cdrt_ref[self.indices_tau_drt_out]
                    dev = np.sum(dev ** 2)
                    return dev

                res = minimize(_loss_alpha, self.rq_pars['alpha'][unfixed_indices], method='TNC',
                               bounds=alpha_bounds_unfixed)
                self.alpha_ref[unfixed_indices] = apply_bounds(res.x, alpha_bounds_unfixed)
                self.rq_pars['alpha'][unfixed_indices] = self.alpha_ref[unfixed_indices]
                self._gamma_gen, _ = self.drt_gen(tau_out, tau_drt, mu, phi)
                self._cdrt_gen = drt_analyzer.cumulative_drt(self._gamma_gen, tau_out)

        if not silent:
            print_to_gui('Fitting with CNLS in EIS...')

        def make_window(tau_vec, tau_pars, element_id):
            # EIS frequency window function, weighting data for element-wise fitting
            lntau_vec = np.log(tau_vec)
            lncenter_tau = np.log(tau_pars[element_id])
            lntau_pars_sorted = np.sort(np.log(tau_pars))
            left_bound = -np.inf
            right_bound = np.inf
            element_rank = np.argmin(np.abs(lntau_pars_sorted - lncenter_tau))
            if 1 <= element_rank:
                left_bound = lntau_pars_sorted[element_rank - 1]
            if element_rank < n - 1:
                right_bound = lntau_pars_sorted[element_rank + 1]

            def smooth_rise(x, a, b):
                a = max(-1e308, min(a, 1e308))
                b = max(-1e308, min(b, 1e308))
                xx = np.maximum(0.0, np.minimum((x - a) / (b - a), 1.0))
                return xx ** 2 * (3 - 2 * xx)

            window = smooth_rise(lntau_vec, left_bound, lncenter_tau)
            window *= smooth_rise(lntau_vec, right_bound, lncenter_tau)
            return window

        if fix_tau:
            if m_ecm_pars_ is None:
                n_unfixed = n  # n = self.n_rq = m_ecm_pars.n
            else:
                n_unfixed = len(unfixed_indices)

            def _loss_fix_tau(R_alpha_vec):  # R is unfixed regardless of m_ecm_pars[*].fixed, while alpha can be fixed.
                Rs = np.maximum(0.0, R_alpha_vec[0: n - 1]) * total_R
                Rs = np.append(Rs, total_R - np.sum(Rs))
                self.rq_pars['R'] = apply_bounds(Rs, R_bounds)
                self.rq_pars['alpha'][unfixed_indices] = apply_bounds(R_alpha_vec[n - 1: n - 1 + n_unfixed],
                                                                      alpha_bounds_unfixed)
                z_gen = self.eis_gen(freq)
                dev = z_gen - z_ref
                dev = dev.real ** 2 + dev.imag ** 2
                dev = np.sum(dev * weights)
                return dev

            rs = self.rq_pars['R'][:-1] / total_R
            rq_par_vec = np.hstack([rs, self.rq_pars['alpha'][unfixed_indices]])
            res = minimize(_loss_fix_tau, rq_par_vec, method='TNC',
                           bounds=merge_bounds(indexed_bounds(R_bounds, np.arange(n - 1)),
                                               alpha_bounds_unfixed))
            rq_par_vec = res.x
            Rs = np.maximum(0.0, rq_par_vec[0: n - 1]) * total_R
            Rs = np.append(Rs, max(total_R - np.sum(Rs), 0.0))
            self.rq_pars['R'] = apply_bounds(Rs, R_bounds)
            self.rq_pars['alpha'][unfixed_indices] = apply_bounds(rq_par_vec[n - 1: n - 1 + n_unfixed],
                                                                  alpha_bounds_unfixed)

            if fit_elementwise:  # fit element-wise after fitting all elements concurrently
                def fit_one_element_fix_tau(element_id):
                    element_window = make_window(1 / (2 * np.pi * freq), self.rq_pars['tau'], element_id)

                    def _loss_one_element_fix_tau(var):
                        self.rq_pars['R'][element_id] = \
                            max(R_bounds.lb[element_id], min(var[0], R_bounds.ub[element_id]))
                        self.rq_pars['alpha'][element_id] = \
                            max(alpha_bounds.lb[element_id], min(var[1], alpha_bounds.ub[element_id]))
                        z_gen = self.eis_gen(freq)
                        dev = z_gen - z_ref
                        dev = dev.real ** 2 + dev.imag ** 2
                        dev = np.sum(dev * weights * element_window)
                        return dev

                    r_alpha_vec = np.array([self.rq_pars['R'][element_id], self.rq_pars['alpha'][element_id]])
                    res = minimize(_loss_one_element_fix_tau, r_alpha_vec, method='TNC',
                                   bounds=Bounds([R_bounds.lb[element_id],
                                                  alpha_bounds.lb[element_id]],
                                                 [R_bounds.ub[element_id],
                                                  alpha_bounds.ub[element_id]]))
                    r_alpha_vec = res.x
                    self.rq_pars['R'][element_id] = \
                        max(R_bounds.lb[element_id], min(r_alpha_vec[0], R_bounds.ub[element_id]))
                    self.rq_pars['alpha'][element_id] = \
                        max(alpha_bounds.lb[element_id], min(r_alpha_vec[1], alpha_bounds.ub[element_id]))

                for element_id in unfixed_indices:
                    fit_one_element_fix_tau(element_id)
        else:  # not fix_tau
            if m_ecm_pars_ is None:
                n_unfixed = n  # n = self.n_rq = m_ecm_pars.n
            else:
                n_unfixed = len(unfixed_indices)

            def _loss(rq_par_vec):
                Rs = np.maximum(0.0, rq_par_vec[0: n - 1]) * total_R
                Rs = np.append(Rs, total_R - np.sum(Rs))
                self.rq_pars['R'] = apply_bounds(Rs, R_bounds)
                self.rq_pars['tau'][unfixed_indices] = \
                    apply_bounds(rq_par_vec[n - 1: n - 1 + n_unfixed], tau_bounds_unfixed)
                self.rq_pars['alpha'][unfixed_indices] = \
                    apply_bounds(rq_par_vec[n - 1 + n_unfixed: n - 1 + n_unfixed * 2], alpha_bounds_unfixed)
                z_gen = self.eis_gen(freq)
                dev = z_gen - z_ref
                dev = dev.real ** 2 + dev.imag ** 2
                dev = np.sum(dev * weights)
                return dev

            if not hasattr(self.rq_pars['R'], '__len__'):
                self.rq_pars['R'] = [self.rq_pars['R']]
            rs = self.rq_pars['R'][:-1] / total_R
            rq_par_vec = np.hstack([rs, self.rq_pars['tau'][unfixed_indices], self.rq_pars['alpha'][unfixed_indices]])
            res = minimize(_loss, rq_par_vec, method='TNC', bounds=merge_bounds(indexed_bounds(R_bounds,
                                                                                               np.arange(n - 1)),
                                                                                merge_bounds(tau_bounds_unfixed,
                                                                                             alpha_bounds_unfixed)))
            rq_par_vec = res.x
            Rs = np.maximum(0.0, rq_par_vec[0: n - 1]) * total_R
            Rs = np.append(Rs, max(total_R - np.sum(Rs), 0.0))
            self.rq_pars['R'] = apply_bounds(Rs, R_bounds)
            self.rq_pars['tau'][unfixed_indices] = \
                apply_bounds(rq_par_vec[n - 1: n - 1 + n_unfixed], tau_bounds_unfixed)
            self.rq_pars['alpha'][unfixed_indices] = \
                apply_bounds(rq_par_vec[n - 1 + n_unfixed: n - 1 + n_unfixed * 2], alpha_bounds_unfixed)

            if fit_elementwise:
                def fit_one_element(element_id):
                    element_window = make_window(1 / (2 * np.pi * freq), self.rq_pars['tau'], element_id)

                    def _loss_one_element(var):
                        self.rq_pars['R'][element_id] = \
                            max(R_bounds.lb[element_id], min(var[0], R_bounds.ub[element_id]))
                        self.rq_pars['tau'][element_id] = \
                            max(tau_bounds.lb[element_id], min(np.exp(var[1]), tau_bounds.ub[element_id]))
                        self.rq_pars['alpha'][element_id] = \
                            max(alpha_bounds.lb[element_id], min(var[2], alpha_bounds.ub[element_id]))
                        z_gen = self.eis_gen(freq)
                        dev = z_gen - z_ref
                        dev = dev.real ** 2 + dev.imag ** 2
                        dev = np.sum(dev * weights * element_window)
                        return dev

                    par_vec = np.array([self.rq_pars['R'][element_id], np.log(self.rq_pars['tau'][element_id]),
                                        self.rq_pars['alpha'][element_id]])
                    res = minimize(_loss_one_element, par_vec, method='TNC',
                                   bounds=Bounds([R_bounds.lb[element_id], np.log(tau_bounds.lb[element_id]),
                                                  alpha_bounds.lb[element_id]],
                                                 [R_bounds.ub[element_id], np.log(tau_bounds.ub[element_id]),
                                                  alpha_bounds.ub[element_id]]))
                    par_vec = res.x
                    self.rq_pars['R'][element_id] = \
                        max(R_bounds.lb[element_id], min(par_vec[0], R_bounds.ub[element_id]))
                    self.rq_pars['tau'][element_id] = \
                        max(tau_bounds.lb[element_id], min(np.exp(par_vec[1]), tau_bounds.ub[element_id]))
                    self.rq_pars['alpha'][element_id] = \
                        max(alpha_bounds.lb[element_id], min(par_vec[2], alpha_bounds.ub[element_id]))

                for element_id in unfixed_indices:
                    fit_one_element(element_id)

        if not silent:
            print_to_gui('Fitting done.')
        return

    def estimate_Rs(self, gamma, cdrt, peak_indices):
        if cdrt[-1] < cdrt[0]:
            cdrt = np.flip(cdrt)
            gamma = np.flip(gamma)
            # peak_indices = cdrt.size - 1 - peak_indices
        if len(peak_indices) < 2:
            return cdrt[-1] - cdrt[0], np.array([0, cdrt.size - 1])
        rank_peak_indices = np.argsort(peak_indices)
        sorted_peak_indices = np.sort(peak_indices)
        inv_peak_vals = 1.0 + 0.0 / (gamma[sorted_peak_indices] + 1e-100)  # weighting inactive
        plateu_indices = np.array((sorted_peak_indices[1:] * inv_peak_vals[1:] +
                                   sorted_peak_indices[:-1] * inv_peak_vals[:-1]) /
                                  (inv_peak_vals[1:] + inv_peak_vals[:-1]) + 0.499, dtype=int)  # mid-way between peaks
        cRs = cdrt[plateu_indices]
        plateu_indices = np.append(plateu_indices, cdrt.size - 1)
        plateu_indices = np.insert(plateu_indices, 0, 0)
        cRs = np.append(cRs, np.max(cdrt))
        cRs = np.insert(cRs, 0, 0.0)
        Rs = cRs[1:] - cRs[:-1]
        Rs = np.array(Rs[np.argsort(rank_peak_indices)])
        return Rs, plateu_indices

    def estimate_alphas(self, gamma, peak_indices, Rs):
        peak_vals = gamma[peak_indices]
        rel_peak_height = peak_vals / Rs * 2 * np.pi
        correction_factor = 0.9  # correctes the limited peak height in RBF-based DRT
        alphas = np.arctan(rel_peak_height * correction_factor) / correction_factor / (np.pi / 2)
        return alphas

    def fit_cdrt(self, freq, zre, zim, silent=False, fit_coupled=False,
                 fit_R=True, fit_tau=True, fit_alpha=True, iter_fit=1, gerischer_ids=None, inherit=False,
                 eis_significance=None, m_ecm_pars=None, R_inf_set=None, L_set=None):
        # fit with cumulative distribution of relaxation time
        if self.cdrt_ref is None or self._cdrt_gen is None or self.gamma_ref is None or \
                self._gamma_gen is None or self.tau_out is None or self.eis_weights is None:
            inherit = False

        if eis_significance is None:
            eis_significance = np.ones(len(freq))

        if not silent:
            print_to_gui('Evaluating DRT...')
        drt_analyzer = util_tikhonov.DRTAnalyzer()
        drt_analyzer.settings = self.drt_settings
        L, R_inf, gamma_ref, tau_out, phi, weights = drt_analyzer.drt_iterative(freq, zre, zim,
                                                                                eis_significance=eis_significance,
                                                                                R_inf_set=R_inf_set, L_set=L_set)
        self._L_ref = L
        self.R_inf_ref = R_inf
        self.gamma_ref = gamma_ref
        self.tau_out = tau_out
        self.phi = phi
        self.eis_weights = weights
        mu = drt_analyzer.mu
        tau_drt = drt_analyzer.tau_drt
        cdrt_ref = drt_analyzer.cumulative_drt(gamma_ref, tau_out)
        # from util_plot import plot_xy, show_plot
        # hdl = plot_xy(tau_out, cdrt_ref, xlog=True, xlabel=r'$\tau$' + ' / s', ylabel='cDRT')
        # show_plot(hdl)
        self.cdrt_ref = cdrt_ref
        total_R = cdrt_ref[-1] - cdrt_ref[0]
        tau_min = np.min(tau_drt) * (1 + 1e-10)
        tau_max = np.max(tau_drt) * (1 - 1e-10)

        if isinstance(m_ecm_pars, ManualECMPars):
            m_ecm_pars_ = ManualECMPars()
            m_ecm_pars_.from_str(str(m_ecm_pars))
            m_ecm_pars_.sort_tau(reverse=True)
            gerischer_ids = gerischer_pos_to_id(m_ecm_pars_.get_gerischer_indices() * -1)
            tau_bounds = m_ecm_pars_.get_tau_bounds()
            tau_bounds = Bounds(tau_bounds[0], tau_bounds[1])
            unfixed_indices = m_ecm_pars_.get_unfixed_indices() - 1
            tau_bounds, unfixed_indices = \
                limit_bounds_and_unfixed_indices(tau_bounds, unfixed_indices,
                                                 tau_min=np.min(tau_drt), tau_max=np.max(tau_drt))
            alpha_bounds = m_ecm_pars_.get_alpha_bounds()
            alpha_bounds = Bounds(alpha_bounds[0], alpha_bounds[1])
            self.n_rq = m_ecm_pars_.n
            taus = np.sqrt(tau_bounds.lb * tau_bounds.ub)
            self.rq_pars['tau'] = taus * 1
            self.rq_pars['R'] = self.estimate_Rs(gamma_ref, cdrt_ref, find_nearest_indices(tau_out, taus))[0]
            if not inherit:
                self.alpha_ref = (alpha_bounds.lb + alpha_bounds.ub * 9) / 10
                self.rq_pars['alpha'] = self.alpha_ref
        else:
            if not silent:
                print_to_gui('Estimating tau...')
            peak_indices = self.find_gamma_peaks(gamma_ref, tau_out, max_n_peaks=self.n_rq)
            self.n_rq = peak_indices.size
            tau_bounds = Bounds([tau_min, ] * self.n_rq, [tau_max, ] * self.n_rq)
            unfixed_indices = np.arange(self.n_rq)
            alpha_bounds = Bounds([0.4, ] * self.n_rq, [1.0, ] * self.n_rq)
            tau_bounds, unfixed_indices = \
                limit_bounds_and_unfixed_indices(tau_bounds, unfixed_indices,
                                                 tau_min=np.min(tau_drt), tau_max=np.max(tau_drt))
            self.rq_pars['R'] = self.estimate_Rs(gamma_ref, cdrt_ref, peak_indices)[0]
            self.rq_pars['tau'] = apply_bounds(tau_out[peak_indices], tau_bounds)
            if not inherit:
                self.alpha_ref = np.ones(self.n_rq) * 0.999
                self.rq_pars['alpha'] = self.alpha_ref
        R_bounds = Bounds([0, ] * self.n_rq, [float('inf'), ] * self.n_rq)  # not fully effective
        # R_bounds_unfixed = *** not applicable ***
        tau_bounds_unfixed = indexed_bounds(tau_bounds, unfixed_indices)
        alpha_bounds_unfixed = indexed_bounds(alpha_bounds, unfixed_indices)
        if gerischer_ids is not None and gerischer_ids:  # is not empty
            self.gerischer_ids = np.array(gerischer_ids) % self.n_rq
        else:
            self.gerischer_ids = []

        print_to_gui('The fitting process can take up to a few seconds.')
        if fit_coupled:
            n_unfixed = len(unfixed_indices)

            def _loss(rq_par_vec):
                n = self.n_rq
                self.rq_pars['R'] = apply_bounds(rq_par_vec[0: n], R_bounds)
                self.rq_pars['tau'][unfixed_indices] = apply_bounds(rq_par_vec[n: n + n_unfixed], tau_bounds_unfixed)
                self.rq_pars['alpha'][unfixed_indices] = np.maximum(0.4, np.minimum(rq_par_vec[n + n_unfixed: n + 2 * n_unfixed], 1.0))
                gamma_gen, _ = self.drt_gen(tau_out, tau_drt, mu, phi)  # speed limiting step
                self._gamma_gen = gamma_gen
                cdrt_gen = drt_analyzer.cumulative_drt(gamma_gen, tau_out)
                self._cdrt_gen = cdrt_gen
                dev = cdrt_gen - cdrt_ref
                dev = np.sum(dev ** 2)
                return dev

            print_to_gui('Fitting with coupled CNLS in cDRT...')
            rq_par_vec = np.hstack([self.rq_pars['R'], self.rq_pars['tau'], self.rq_pars['alpha']])
            res = minimize(_loss, rq_par_vec, method='TNC', bounds=Bounds(0, np.inf))
            rq_par_vec = res.x
            n = self.n_rq
            self.rq_pars['R'] = apply_bounds(rq_par_vec[0: n], R_bounds)
            self.rq_pars['tau'] = np.maximum(tau_min, np.minimum(rq_par_vec[n: 2 * n], tau_max))
            self.rq_pars['alpha'] = np.maximum(0.4, np.minimum(rq_par_vec[2 * n: 3 * n], 1.0))

        else:  # fit segregated (not coupled)
            for i in range(iter_fit):
                if fit_R:
                    if not silent:
                        print_to_gui('Fitting R...')
                    if not inherit:
                        self.rq_pars['alpha'] = self.rq_pars['alpha'] * 0 + 0.99  # so that the fitting is faster

                    def _loss_R(Rs):
                        self.rq_pars['R'] = apply_bounds(Rs, R_bounds)
                        gamma_gen, _ = self.drt_gen(tau_out, tau_drt, mu, phi)  # speed limiting step
                        self._gamma_gen = gamma_gen
                        cdrt_gen = drt_analyzer.cumulative_drt(gamma_gen, tau_out)
                        self._cdrt_gen = cdrt_gen
                        dev = cdrt_gen - cdrt_ref
                        dev = np.average(dev ** 2) + (np.sum(Rs) - total_R) ** 2
                        return dev

                    res = minimize(_loss_R, self.rq_pars['R'], method='TNC', bounds=Bounds(0, np.inf))
                    self.rq_pars['R'] = res.x

                if fit_tau:
                    if not silent:
                        print_to_gui('Fitting tau...')
                    if not inherit:
                        # so that the fitting is faster
                        self.rq_pars['alpha'][unfixed_indices] = \
                            apply_bounds(self.rq_pars['alpha'][unfixed_indices] * 0 + 0.99, alpha_bounds_unfixed)
                    r_smpl = np.linspace(0, min(total_R, np.sum(self.rq_pars['R'])), 200)
                    lntau_out = np.log(tau_out)
                    inv_cdrt = interp1d(cdrt_ref, lntau_out)
                    lntau_r_smpl_ref = inv_cdrt(r_smpl)  # fit using the inverse function of cDRT (r in, tau out)
                    r_margin = 1e-3

                    def _loss_tau(taus):
                        self.rq_pars['tau'][unfixed_indices] = apply_bounds(taus, tau_bounds_unfixed)
                        gamma_gen, _ = self.drt_gen(tau_out, tau_drt, mu, phi)
                        self._gamma_gen = gamma_gen
                        cdrt_gen = drt_analyzer.cumulative_drt(gamma_gen, tau_out)
                        self._cdrt_gen = cdrt_gen
                        inv_cdrt = interp1d(cdrt_gen, lntau_out)
                        if lntau_out[0] < lntau_out[-1]:
                            min_r = cdrt_gen[0] * (1 - r_margin) + cdrt_gen[-1] * r_margin
                            max_r = cdrt_gen[-1] * (1 - r_margin) + cdrt_gen[0] * r_margin
                        else:
                            max_r = cdrt_gen[0] * (1 - r_margin) + cdrt_gen[-1] * r_margin
                            min_r = cdrt_gen[-1] * (1 - r_margin) + cdrt_gen[0] * r_margin
                        r_smpl_lim = np.maximum(min_r, np.minimum(r_smpl, max_r))
                        dev = inv_cdrt(r_smpl_lim) - lntau_r_smpl_ref
                        dev = np.sum(dev ** 2)
                        return dev

                    res = minimize(_loss_tau, self.rq_pars['tau'][unfixed_indices], method='TNC',
                                   bounds=tau_bounds_unfixed)
                    self.rq_pars['tau'][unfixed_indices] = apply_bounds(res.x, tau_bounds_unfixed)

                if fit_alpha:
                    if not silent:
                        print_to_gui('Fitting alpha...')
                    if not inherit:
                        # so that the fitting is faster
                        self.rq_pars['alpha'][unfixed_indices] = \
                            apply_bounds(self.rq_pars['alpha'][unfixed_indices] * 0 + 0.99, alpha_bounds_unfixed)

                    def _loss_alpha(alphas):
                        self.rq_pars['alpha'][unfixed_indices] = apply_bounds(alphas, alpha_bounds_unfixed)
                        gamma_gen, _ = self.drt_gen(tau_out, tau_drt, mu, phi)
                        self._gamma_gen = gamma_gen
                        cdrt_gen = drt_analyzer.cumulative_drt(gamma_gen, tau_out)
                        self._cdrt_gen = cdrt_gen
                        dev = cdrt_gen - cdrt_ref
                        dev = np.sum(dev ** 2)
                        return dev

                    res = minimize(_loss_alpha, self.rq_pars['alpha'][unfixed_indices], method='TNC',
                                   bounds=alpha_bounds_unfixed)
                    self.rq_pars['alpha'][unfixed_indices] = apply_bounds(res.x, alpha_bounds_unfixed)

        self._gamma_gen_indiv, _ = self.drt_gen(tau_out, tau_drt, mu, phi, indiv_elements=True)
        if (not fit_R) and (not fit_tau) and (not fit_alpha):
            gamma_gen, _ = self.drt_gen(tau_out, tau_drt, mu, phi)
            self._gamma_gen = gamma_gen
            cdrt_gen = drt_analyzer.cumulative_drt(gamma_gen, tau_out)
            self._cdrt_gen = cdrt_gen

        if not silent:
            print_to_gui('Fitting done.')
        return

    @staticmethod
    def _drt_sim_gerischer(z0, tau0, tau_out):
        isingularity = 0
        n = len(tau_out)
        increament_flag = True
        lntaus = np.log(tau_out)
        dlntau = np.average(lntaus[1:] - lntaus[:-1])
        if dlntau < 0:
            increament_flag = False
            dlntau = -dlntau
        half_step = np.exp(dlntau / 2)
        tau_out_copy = 1 * tau_out
        if not increament_flag:
            tau_out_copy = np.flip(tau_out_copy)

        for i in range(1, n-1):
            if (tau_out_copy[i] - tau0) * (tau_out_copy[i-1] - tau0) < 0 or tau_out_copy[i] == tau0:
                isingularity = i - 1
                break

        def averaged_drt_std_gerischer(tau0, tau1, tau2):
            a = min(1.0, tau1 / tau0)
            b = min(1.0, tau2 / tau0)
            if a == b == 1:
                return 0.0
            if a < 0 or b < 0 or a == b:
                raise ValueError('Either tau0, tau1 or tau2 is negative, or tau1 == tau2')
            # assumption: b > a >= 0
            I = 2 / np.pi * (np.arccos(np.sqrt(a)) - np.arccos(np.sqrt(b)))
            return I / (tau2 - tau1) * tau0

        def _drt_gerischer(z0, tau0, tau=None):
            if tau is None:
                tau = tau_out
            frac = tau / (tau0 - tau)
            frac = np.maximum(0, frac)
            imp = np.sqrt(frac) / np.pi
            imp = np.where(imp == np.inf, 0, imp)
            return z0 * imp

        drt_target = np.zeros(n)
        drt_target[isingularity + 1] = averaged_drt_std_gerischer(tau0, tau_out_copy[isingularity + 1] / half_step,
                                                                  tau_out_copy[isingularity + 1] * half_step) * z0
        drt_target[isingularity] = averaged_drt_std_gerischer(tau0, tau_out_copy[isingularity] / half_step,
                                                              tau_out_copy[isingularity] * half_step) * z0
        drt_target[isingularity - 1] = averaged_drt_std_gerischer(tau0, tau_out_copy[isingularity - 1] / half_step,
                                                                  tau_out_copy[isingularity - 1] * half_step) * z0
        drt_target[:isingularity - 1] = _drt_gerischer(z0, tau0, tau=tau_out_copy[:isingularity - 1])
        if not increament_flag:
            drt_target = np.flip(drt_target)
        return drt_target

    def drt_gen(self, tau_out, tau_drt, mu, phi, indiv_elements=False):
        # tau_out, tau_drt, mu, phi are all fields of util_tikhonov.DRTAnalyzer instances after running drt_iterative
        r = np.sqrt(np.log(2))
        alpha_crit = fsolve(lambda n: np.cosh(n * r / mu) - (2 + np.cos(n * np.pi)), np.array(0.95))[0]
        tau_drt = np.array(tau_drt)
        tau_out = np.array(tau_out)

        def indexable(x):
            if not hasattr(x, '__len__'):
                x = np.array([x, ])
            return x

        self.rq_pars['R'] = indexable(self.rq_pars['R'])
        self.rq_pars['tau'] = indexable(self.rq_pars['tau'])
        self.rq_pars['alpha'] = indexable(self.rq_pars['alpha'])

        if indiv_elements:
            gamma = np.zeros([self.n_rq, tau_out.size]) * 1j
            for i in range(self.n_rq):
                if i in self.gerischer_ids:
                    gamma[i] = self._drt_sim_gerischer(self.rq_pars['R'][i], self.rq_pars['tau'][i], tau_out=tau_out)
                else:
                    gamma[i] = self._drt_sim_rq(self.rq_pars['R'][i], self.rq_pars['tau'][i], self.rq_pars['alpha'][i],
                                                alpha_crit=alpha_crit, tau_drt=tau_drt, phi=phi, mu=mu, tau_out=tau_out)
        else:
            gamma = np.zeros(tau_out.size) * 1j
            for i in range(self.n_rq):
                if i in self.gerischer_ids:
                    gamma += self._drt_sim_gerischer(self.rq_pars['R'][i], self.rq_pars['tau'][i], tau_out=tau_out)
                else:
                    gamma += self._drt_sim_rq(self.rq_pars['R'][i], self.rq_pars['tau'][i], self.rq_pars['alpha'][i],
                                              alpha_crit=alpha_crit, tau_drt=tau_drt, phi=phi, mu=mu, tau_out=tau_out)
        gamma = gamma.real
        return gamma, tau_out

    @staticmethod
    def z_gerischer(omega, z0, tau):
        return z0 / np.sqrt(1 + 1j * omega * tau)

    @staticmethod
    def eis_indiv_cummulative(eis_indiv, R_inf, sequence=None, hi_f_to_lo_f=True):
        if not hasattr(eis_indiv[0], '__len__'):
            print_to_gui('eis_indiv_cummulative: eis_indiv contains only 1 array')
            return np.array(eis_indiv) + R_inf
        if sequence is None or len(sequence) != len(eis_indiv):
            sequence = np.flip(np.arange(len(eis_indiv)))
        if hi_f_to_lo_f:
            end_index = -1
        else:
            end_index = 0
        cR = R_inf + 0j
        for i in sequence:
            R = eis_indiv[i][end_index]
            eis_indiv[i] = np.array(eis_indiv[i]) + cR
            cR += R
        return eis_indiv

    def eis_gen(self, freq, indiv_elements=False):
        omega = 2 * np.pi * np.array(freq)
        gerischer_ids = self.gerischer_ids
        if indiv_elements:
            z = np.zeros([self.n_rq, omega.size]) * 1j
            for i in range(self.n_rq):
                if i in gerischer_ids:
                    z[i] = self.z_gerischer(omega, self.rq_pars['R'][i], self.rq_pars['tau'][i])
                else:
                    z[i] = self.z_rq(omega, self.rq_pars['R'][i], self.rq_pars['tau'][i], self.rq_pars['alpha'][i])
                # z[i] += self.R_inf_ref
                # if hasattr(self.L_ref, '__len__'):
                #     z[i] += (self.L_ref[0] * 1j + self.L_ref[1]) * 2 * np.pi * freq
                # else:
                #     z[i] += self.L_ref * 2j * np.pi * freq
        else:
            z = np.zeros(omega.size) * 1j
            for i in range(self.n_rq):
                if i in gerischer_ids:
                    z += self.z_gerischer(omega, self.rq_pars['R'][i], self.rq_pars['tau'][i])
                else:
                    z += self.z_rq(omega, self.rq_pars['R'][i], self.rq_pars['tau'][i], self.rq_pars['alpha'][i])
            z += self.R_inf_ref
            if hasattr(self._L_ref, '__len__'):
                z += (self._L_ref[0] * 1j + self._L_ref[1]) * 2 * np.pi * freq
            else:
                z += self._L_ref * 2j * np.pi * freq
        return z

    def find_gamma_peaks(self, gamma, tau, sup_hi=0.0, sup_lo=0.0, max_n_peaks=7, thresh=1e-2, damp_dec=None):
        if damp_dec is None:
            damp_dec = self.peak_damp_dec
        if tau[0] < tau[-1]:  # swap
            sup_lo, sup_hi = sup_hi, sup_lo
        sup_hi = max(0.0, min(2 - 2 * np.sqrt(sup_hi), 2.0))
        sup_lo = max(-1.0, min(2 * np.sqrt(sup_lo) - 1, 1.0))

        def logistic(x):
            return 1 / (1 + np.exp(-8 * x))

        x = np.linspace(0, 1, len(tau) - 2)  # without head and tail
        weights = logistic(sup_hi - x) * logistic(x - sup_lo)

        x = np.log(tau)
        x -= (np.max(x) + np.min(x)) / 2
        damp_dec *= np.log(10)
        kernel = -np.exp(-x ** 2 / (2 * damp_dec ** 2)) / (np.sqrt(2 * np.pi) * damp_dec ** 3) * \
                 (x ** 2 / damp_dec ** 2 - 1)
        d2gamma = convolve(gamma, kernel, mode='same')
        d2gamma = d2gamma / np.max(d2gamma)

        def compress(x, order=3):
            return 1 - np.power(1 - x, order)

        sel_gamma = compress(d2gamma, order=3) * 0.9 + d2gamma * 0.1
        sel_gamma = sel_gamma * 0.9 + gamma / np.max(gamma) * 0.1
        sel_gamma = np.maximum(0.0, sel_gamma)

        # import util_plot
        # hdl = util_plot.plot_xy(tau, gamma / np.max(gamma), xlog=True)
        # hdl = util_plot.plot_xy(tau, sel_gamma, xlog=True, hdl=hdl)
        # util_plot.show_plot(hdl)

        thresh_prominence = thresh * np.max(sel_gamma)
        peak_res = find_peaks(sel_gamma, prominence=thresh_prominence)
        peak_indices = peak_res[0]
        peak_prominences = peak_res[1]['prominences'] * weights[peak_indices]

        peak_count = min(max_n_peaks, len(peak_indices))
        peak_ranks = np.argsort(peak_prominences)  # ascending order
        ranked_peak_indices = peak_indices[peak_ranks[-peak_count:]]
        ranked_peak_indices = np.array(ranked_peak_indices, dtype=np.int)
        ranked_peak_indices = np.sort(ranked_peak_indices)
        if tau[0] < tau[-1]:
            ranked_peak_indices = np.flip(ranked_peak_indices)
        return ranked_peak_indices

    # @staticmethod
    # def find_gamma_peaks_old(gamma, tau, sup_hi=0.0, sup_lo=0.0, max_n_peaks=6, thresh=1e-2, min_interpeak_dec=0.2):
    #     if tau[0] < tau[-1]:  # swap
    #         sup_lo, sup_hi = sup_hi, sup_lo
    #     sup_hi = max(0.0, min(2 - 2 * np.sqrt(sup_hi), 2.0))
    #     sup_lo = max(-1.0, min(2 * np.sqrt(sup_lo) - 1, 1.0))
    #
    #     def logistic(x):
    #         return 1 / (1 + np.exp(-8 * x))
    #
    #     x = np.linspace(0, 1, len(tau) - 2)  # without head and tail
    #     weights = logistic(sup_hi - x) * logistic(x - sup_lo)
    #
    #     lntau = np.log(tau)
    #     d1gamma = (gamma[1:] - gamma[:-1]) / (lntau[1:] - lntau[:-1])
    #     d2gamma = -(d1gamma[1:] - d1gamma[:-1]) / (lntau[2:] - lntau[:-2]) * 2
    #
    #     def compress(x, order=3):
    #         return 1 - np.power(1 - x, order)
    #
    #     sel_gamma = compress(d2gamma / np.max(d2gamma), order=3)
    #                 # + 0.2 * compress(gamma[1:-1] / np.max(gamma[1:-1]), order=2)
    #     sel_gamma = np.maximum(0.0, sel_gamma)
    #     # import util_plot
    #     # hdl = util_plot.plot_xy(tau, gamma / np.max(gamma), xlog=True)
    #     # hdl = util_plot.plot_xy(tau[1:-1], sel_gamma, xlog=True, hdl=hdl)
    #     # util_plot.show_plot(hdl)
    #     thresh_prominence = thresh * np.max(sel_gamma)
    #     peak_res = find_peaks(sel_gamma, prominence=thresh_prominence)
    #     peak_indices = peak_res[0]
    #     peak_prominences = peak_res[1]['prominences'] * weights[peak_indices]
    #
    #     def arg_merge_peaks(peak_lntaus, min_interpeak_dec):
    #         # peak_lntaus is assumed sorted in either order
    #         peak_lgtau = np.array(peak_lntaus) / np.log(10)
    #         pairs_to_merge = []  # list of pairs of indices
    #         # list of lists of a single number: i.e. [[0], [1], [4], ...]
    #         n = len(peak_lgtau)
    #         isolated_indices = [True] * n  # list of indices that are not paired
    #
    #         # find pairs of peaks to merge
    #         for i in range(n - 1):
    #             if np.abs(peak_lgtau[i + 1] - peak_lgtau[i]) < min_interpeak_dec:
    #                 pairs_to_merge.append([i, i + 1])
    #                 isolated_indices[i] = False
    #                 isolated_indices[i + 1] = False
    #         # if not pairs_to_merge == if pairs_to_merge is empty
    #         isolated_indices = np.argwhere(isolated_indices).T[0].tolist()
    #
    #         # merge adjacent pairs into peak groups
    #         groups_to_merge = []  # list of groups of indices
    #         if pairs_to_merge:  # if not empty
    #             groups_to_merge.append(pairs_to_merge[0])
    #             for pair in pairs_to_merge:
    #                 if groups_to_merge[-1][-1] == pair[0]:
    #                     # if the last element of the group matches the first element of the pair
    #                     groups_to_merge[-1] += pair[1:]  # merge the pair into the group
    #                     # pair[1:] is a list while pair[1] is a number, even if len(pair) == 2
    #                 else:
    #                     groups_to_merge.append(pair[0:])  # pair[0:] is a deep copy of pair
    #             groups_to_merge = groups_to_merge[1:]  # remove the first element which duplicates
    #         groups_to_merge += isolated_indices
    #         return groups_to_merge  # groups of indices of peaks in peak_lntaus
    #
    #     prominence_weights = weights[peak_indices]
    #     peak_prominences = peak_prominences * prominence_weights
    #
    #     peak_lntaus = lntau[peak_indices]
    #     grouped_peak_indices = arg_merge_peaks(peak_lntaus, min_interpeak_dec)
    #     averaged_peak_indices = np.zeros(len(grouped_peak_indices))
    #     merged_peak_prominences = np.zeros(len(grouped_peak_indices))
    #
    #     for i, group in enumerate(grouped_peak_indices):
    #         average_index = int(0.4999 + np.average(peak_indices[group]))
    #         # if int(0.5 + sth), there's a risk of index overflow
    #         averaged_peak_indices[i] = average_index
    #         merged_peak_prominences[i] = np.max(peak_prominences[group])
    #
    #     peak_count = min(max_n_peaks, len(averaged_peak_indices))
    #     peak_ranks = np.argsort(merged_peak_prominences)  # ascending order
    #     ranked_peak_indices = averaged_peak_indices[peak_ranks[-peak_count:]]
    #     ranked_peak_indices = np.array(ranked_peak_indices, dtype=np.int) + 1
    #     ranked_peak_indices = np.sort(ranked_peak_indices)
    #     if tau[0] < tau[-1]:
    #         ranked_peak_indices = np.flip(ranked_peak_indices)
    #     return ranked_peak_indices

    @staticmethod
    def z_rq(omega, z0, tau, alpha):
        if alpha < 0 or alpha > 1:
            raise ValueError('Improper alpha value for RQ element!')
        if alpha == 0:
            return z0 / 2
        return z0 / (1 + np.power(1j * omega * tau, alpha))

    def _drt_sim_rc(self, z0, tau0, tau_drt, phi, mu, fast=False):
        if fast:
            phi = phi[self.indices_tau_drt_out]
        if tau0 > np.max(tau_drt) or tau0 < np.min(tau_drt):
            return phi @ (0.0 * tau_drt)
            # raise ValueError('Internal bug: tau0 does not lie within the range of tau_drt!')

        increasing_tau = True  # tau_drt grows with index i.e. tau_drt[i+1] > tau_drt[i]
        dlntau = np.log(tau_drt[-1] / tau_drt[0]) / (len(tau_drt) - 1)
        if dlntau < 0:
            increasing_tau = False
            dlntau = -dlntau
        rel_smpl_interval = dlntau * mu * np.sqrt(2)
        n = len(tau_drt)
        x = np.zeros(n)
        lntau0 = np.log(tau0)
        lntaudrt = np.log(tau_drt)
        if increasing_tau:
            i = np.sum(lntaudrt < lntau0)
        else:
            i = np.sum(lntaudrt > lntau0)
        d = lntau0 - lntaudrt[i]
        rel_pos = d / (lntaudrt[i - 1] - lntaudrt[i])  # ideally this equals dlntau or -dlntau
        w1, w2 = self._weights(rel_pos, rel_smpl_interval)
        x[i], x[i - 1] = w1 * z0, w2 * z0
        # correction = np.average(np.log(tau_out[1:] / tau_out[:-1])) * np.sum(phi[:, int(phi.shape[1] / 2)])
        correction = dlntau * np.sum(phi[int(phi.shape[0] / 2)])
        gamma = phi @ x / correction
        return gamma

    def _drt_rq(self, z0, tau0, alpha, tau_out):
        tau = tau_out
        if alpha < 0 or alpha > 1:
            raise ValueError('Improper alpha value for RQ element!')
        if alpha == 0:
            alpha = 1e-100
        if alpha == 1:
            alpha = 1.0 - 1e-15
        z = z0 * np.sin(alpha * np.pi) / (2 * np.pi * (self.cosh_scaled(alpha * np.log(tau0 / tau))
                                                       + np.cos(alpha * np.pi)))
        return z.real

    def _drt_sim_rq(self, z0, tau0, alpha, alpha_crit, tau_out, tau_drt, phi, mu):
        if alpha <= 0 or alpha > 1:
            raise ValueError('Improper alpha value for RQ element!')
        fast = (len(tau_out) == len(tau_drt))
        if alpha > alpha_crit:
            drt_pulse = self._drt_sim_rc(z0, tau0, tau_drt, phi, mu, fast=fast)
            drt_crit = self._drt_rq(z0, tau0, alpha_crit, tau_out)
            r = (alpha - alpha_crit) / (1 - alpha_crit)
            return drt_pulse * r + drt_crit * (1 - r)
        else:
            drt_target = self._drt_rq(z0, tau0, alpha, tau_out)
            return drt_target

    @staticmethod
    def _weights(rel_pos, rel_smpl_interval):
        # There are 2 gaussian functions centered at a and b with standard deviation sigma = 1/mu
        # Their weighted sum should proximate another gaussian function with the same sigma and centered elsewhere
        # rel_pos = (x - a)/(b - a): the peak position x relative to its neighboring sample points a, b, a <= x <= b
        # rel_smpl_interval = dx * mu: the ratio of sampling interval dx over standard deviation sigma = 1/mu
        if rel_smpl_interval > 1:
            raise ValueError('The RBF is too thin for the given frequency sampling interval! '
                             'Try shape factors <= 0.554 = 1/sqrt( sqrt(2)*ln(10) ), 0.5 recommended.')
        if not 0 <= rel_pos <= 1:
            raise ValueError('The relative position rel_pos must satisfy: 0 <= rel_pos <= 1')
        # x = rel_pos * rel_smpl_interval
        correction = 1  # correction = 1 + 0.5 * x * (rel_smpl_interval - x)
        w1 = (1 - rel_pos) * correction
        w2 = rel_pos * correction
        return w1, w2

    @staticmethod
    def cosh_scaled(x, scale=1e-0, lim=1.5e308):

        # res = 0j * np.array(x)
        # lnlim = np.log(lim)
        # lnscale = np.log(scale)
        # x1 = np.where(np.abs(x.real) > lnlim - lnscale, lim * np.exp(1j * x.imag) * np.sign(x.real), 0)
        # x = np.where(np.abs(x.real) > lnlim - lnscale, 0, x)
        # x2 = np.where((np.abs(x.real) > lnlim) * (np.abs(x.real) <= lnlim - lnscale),
        #               np.exp(np.abs(x.real) + lnscale - np.log(2)) * np.exp(1j * x.imag) * np.sign(x.real), 0)
        # temp = lnlim - max(0.0, lnscale)
        # x = np.where(np.abs(x.real) > temp, 0, x)
        # x3 = np.where(np.abs(x.real) > temp, 0, np.cosh(x) * scale)
        # res += x1 + x2 + x3

        # n = len(x)
        # for i in range(n):
        #     if abs(x[i].real) > 700 - np.log(scale):
        #         res[i] = np.inf * np.exp(1j * x[i].imag) * np.sign(x[i].real)
        #     elif abs(x[i].real) > 700:
        #         res[i] = 0.5 * np.exp(np.abs(x[i]) + np.log(scale))
        #     else:
        #         res[i] = np.cosh(x[i]) * scale

        res = np.cosh(x)
        return res


def main_test_fitting(plot=True):
    ecm = ECM()
    import util_io
    area = 16.0  # cm^2
    from os.path import split, join
    filename = "example_eis.txt"
    eis_data = util_io.load_eis_data(filename)
    eis_data[:, 1:] *= area
    freq = eis_data[:, 0]
    lmd = 0.01
    ecm.drt_settings.lmd = lmd
    ecm.drt_settings.auto_lmd = True
    # ss = True
    # s = False
    # ecm.fit_cdrt(freq, eis_data[:, 1], eis_data[:, 2], fit_coupled=s, fit_R=ss, fit_tau=ss, fit_alpha=ss, iter_fit=1)
    ecm.fit_eis(freq, eis_data[:, 1], eis_data[:, 2])

    # filename = 'FittedParameters.txt'
    # print_to_gui(util_io.save_ecm(filename, ecm))

    if plot:
        import util_plot

        # plot cDRT
        # hdl = util_plot.plot_xy(ecm.tau_out, ecm.cdrt_ref, xlog=True, format_ids=((1, 1, 0),))
        # hdl = util_plot.plot_xy(ecm.tau_out, ecm.cdrt_gen, xlog=True, format_ids=((1, 3, 2),), hdl=hdl)
        # util_plot.show_plot(hdl)

        # plot EIS and DRT
        eis_ecm = ecm.eis_gen(freq)
        drt_analyzer = util_tikhonov.DRTAnalyzer()
        drt_analyzer.settings.lmd = lmd
        L, R_inf, gamma, tau, _, weights = \
            drt_analyzer.drt_iterative(freq, eis_ecm.real, eis_ecm.imag, auto_lmd=True, iter_count=1)

        z_abs = np.sqrt(eis_data[:, 1] ** 2 + eis_data[:, 2] ** 2)
        z_res_ratio_re = -(eis_data[:, 1] - eis_ecm.real) / z_abs
        z_res_ratio_im = -(eis_data[:, 2] - eis_ecm.imag) / z_abs

        eis_format_ids = ((1, 0, 0), (2, 8, 0))
        drt_format_ids = ((2, 0, 0), (2, 8, 1))
        eis_labels = ('Planar SOFC',)
        drt_labels = eis_labels
        eis_z_indiv = ecm.eis_indiv_cummulative(ecm.eis_gen(freq, indiv_elements=True), R_inf=ecm.R_inf_ref)
        hdl = util_plot.plot_eisdrtres([freq, freq], [eis_data[:, 1], eis_ecm.real], [eis_data[:, 2], eis_ecm.imag],
                                    [ecm.gamma_ref, gamma], [tau, tau], drt_show_tau=False,
                                    eis_res_real=z_res_ratio_re, drt_highlights=ecm.find_gamma_peaks(gamma, tau),
                                    eis_res_imag=z_res_ratio_im,
                                    eis_z_indiv=eis_z_indiv, drt_gamma_indiv=ecm._gamma_gen_indiv,
                                    eis_labels=eis_labels, drt_labels=drt_labels, eis_weights=weights,
                                    eis_title='EIS', eis_format_ids=eis_format_ids, eis_highlight=False,
                                    drt_show_LR=False, drt_title='DRT', drt_format_ids=drt_format_ids)
        util_plot.show_plot(hdl)

    print_to_gui('Results saved.')


def main_test_mecm():
    # testing min max setters
    pars = ManualECMPars(2)
    elem = pars.get(1)
    elem.tau = 1e-3
    elem.label = 'P1'
    pars.set(1, elem)
    elem = pars.get(2)
    elem.tau = 3e-3
    elem.label = 'P3'
    pars.set(2, elem)
    elem.tau = 2e-3
    elem.label = 'P2'
    pars.append(elem)
    pars.sort_tau()
    # testing to and from string for ManualECMElement
    elem1 = ManualECMElement()
    elem1._R = 0.4
    elem1._Rmin = 0.5
    string = str(elem1)
    elem.from_str(string)
    # testing to and from string for ManualECMPars
    string = str(pars)
    pp = ManualECMPars()
    pp.from_str(string)
    # testing file io for ManualECMPars
    test_filename = 'M_ECM_test.txt'
    pars.save(test_filename)
    pp.load(test_filename)
    print('done.')

def main_test_ecm_io():
    ecm = ECM()
    ecm.load('example_ecm.txt')
    print(ecm)


if __name__ == '__main__':
    import cProfile
    import pstats
    statfile = 'example_eis.txt'
    cProfile.run(statement='main_test_fitting(plot=False)', filename=statfile)
    p = pstats.Stats(statfile)
    p.sort_stats('cumulative').print_stats(50)
    from os import system
    system("pause")
    
