"""
Call this Python code in command line with your EIS data file and you get the EIS data fitted with ECM,
    ECM = Equivalent Circuit Model (arbitrary number of RQ elements)
    along with the EIS of ECM and the DRT of both input EIS and the fitted ECM, which may be saved as data files
Coded by Hangyue Li, Tsinghua University
version 2.5
"""
import sys
from os import path, makedirs, listdir, system
import traceback
import numpy as np

cd = path.dirname(__file__)
sys.path.append(cd)
import util_io
import util_ecm
import util_plot
import util_tikhonov
sys.path.pop()

connected_to_gui = False
plot_ranges_file = './plot_ranges_eisart.txt'
settings_file = './settings_eisart.txt'


def print_to_gui(string, file=sys.stdout):
    print(string, file=file)
    if connected_to_gui:
        try:
            from EISART_support import print_to_gui
            print_to_gui(string)
        except AttributeError:
            pass


def get_drt_settings(settings):
    drt_settings = util_tikhonov.DRTSettings()
    if settings.kk_test:
        drt_settings.lmd = 0.0
    else:
        drt_settings.lmd = settings.lmd
    drt_settings.freq_extension = (settings.drt_lo_f_extension_decades, settings.drt_hi_f_extension_decades)
    if settings.kk_test:
        drt_settings.shape_factor = 1e1000000  # 64 bit floating point infinity
    else:
        drt_settings.shape_factor = 0.5
    drt_settings.iter_count = settings.weight_iter
    drt_settings.auto_lmd = settings.auto_lmd
    # drt_settings.hyper_lmd_iter = settings.hyper_lmd_iter
    drt_settings.auto_lmd_radius_dec = settings.auto_lmd_radius_dec
    drt_settings.false_peak_suppress = settings.drt_false_peak_suppress
    drt_settings.max_trim = settings.max_trim
    drt_settings.trim_level = settings.flt_level
    drt_settings.trim_unskip = settings.trim_unskip
    return drt_settings


def load_eis_and_trim(input_filename, area=1.0, batch=False, save_fmt='txt', convert_ism=False, eis_convert_outpath='',
                      eis_lo_f_discard_decades=0.0, eis_hi_f_discard_decades=0.0):
    # load EIS data
    eis_data = None
    try:
        eis_data = util_io.load_eis_data(input_filename)
        eis_data[:, 1:3] *= area
    except ValueError:
        if not batch:
            print_to_gui('Unsupported data format!')
            if connected_to_gui:
                from EISART_support import message_set
                message_set('Unsupported data format!')
    if eis_data is None:
        return None, (None, None, None)

    # convert ism
    filename_no_path = path.split(input_filename)[1]
    ext = path.splitext(input_filename)[1]
    if save_fmt == 'csv':
        delim = ','
    else:
        delim = '\t'
    if convert_ism and ext[1:] == 'ism':
        if eis_convert_outpath is None:
            eis_convert_outpath = path.split(input_filename)[0]
        if path.exists(eis_convert_outpath):
            pass
        else:
            makedirs(eis_convert_outpath)
        eis_smooth_name = path.join(eis_convert_outpath, filename_no_path)  # change path
        util_io.save_eis_data(path.splitext(eis_smooth_name)[0] + '.' + save_fmt,
                              eis_data[:, 0], eis_data[:, 1], eis_data[:, 2], delim=delim)

    # discard data according to settings
    n = eis_data.shape[0]
    freq = eis_data[:, 0]
    lo_f_factor = np.power(10.0, eis_lo_f_discard_decades)
    hi_f_factor = np.power(10.0, eis_hi_f_discard_decades)
    if freq[0] < freq[-1]:  # if frequency ranked from low to high, swap
        eis_left_f_discard_count = np.sum(freq < freq[0] * lo_f_factor)
        eis_right_f_discard_count = np.sum(freq > freq[-1] / hi_f_factor)
    else:
        eis_left_f_discard_count = np.sum(freq > freq[0] / hi_f_factor)
        eis_right_f_discard_count = np.sum(freq < freq[-1] * lo_f_factor)
    if eis_right_f_discard_count == 0:
        eis_right_f_discard_count = -n
    eis_data = eis_data[eis_left_f_discard_count: -eis_right_f_discard_count, :]
    return eis_data, (filename_no_path, ext, delim)


def process_one_file_drt(input_filename, batch=False, savedata=True, drt_outpath=None, eis_smooth_outpath=None,
                         hdl=None, area=1.0, settings=None, eis_convert_outpath=None, save_plot_outpath=None,
                         eis_residual_outpath=None, eis_weights_outpath=None, R_inf_set=None, L_set=None):
    # read settings
    if settings is None:
        settings = Settings()
    # load settings
    save_drt_or_smoothed_eis = settings.save_ref_drt and savedata
    convert_ism = settings.convert_ism and savedata
    kk_test = settings.kk_test
    eis_lo_f_discard_decades = settings.eis_lo_f_discard_decades
    eis_hi_f_discard_decades = settings.eis_hi_f_discard_decades
    drt_show_tau = settings.drt_show_tau
    drt_show_LR = settings.drt_show_LR
    eis_draw_lines = settings.eis_draw_lines
    save_fmt = settings.save_fmt
    plot = settings.show_plot
    save_plot = settings.save_plot and savedata
    save_residuals = settings.save_residuals and savedata
    save_weights = settings.save_weights and savedata
    peak_damp_dec = settings.peak_damp_dec

    ret_data = load_eis_and_trim(input_filename, area, batch, save_fmt, convert_ism, eis_convert_outpath,
                                 eis_lo_f_discard_decades, eis_hi_f_discard_decades)
    eis_data, (filename_no_path, ext, delim) = ret_data
    if eis_data is None:
        return hdl

    # read and analyze EIS data
    import util_tikhonov
    freq, z_re, z_im = eis_data.T[:3]  # expected format
    if eis_data.shape[1] > 3:
        # elapsed_time = eis_data.T[3]
        eis_significance = eis_data.T[4]
    else:
        eis_significance = np.ones(len(freq))

    drt_settings = get_drt_settings(settings)
    # drt_settings.sup_L = [0.0, 1e6]
    drt_analyzer = util_tikhonov.DRTAnalyzer()
    drt_analyzer.settings = drt_settings
    L, R_inf, gamma, tau, _, eis_weights = \
        drt_analyzer.drt_iterative(freq, z_re, z_im, eis_significance=eis_significance,
                                   R_inf_set=R_inf_set, L_set=L_set)
    eis_reg = drt_analyzer.eis_from_drt(L, R_inf, gamma, tau, freq)
    eis_noL = drt_analyzer.eis_from_drt(0, R_inf, gamma, tau, freq)  # EIS without inductance
    z_abs = np.sqrt(z_re ** 2 + z_im ** 2)
    z_res_ratio_re = -(z_re - eis_reg.real) / z_abs
    z_res_ratio_im = -(z_im - eis_reg.imag) / z_abs

    # calculate R_0
    R_0 = R_inf + drt_analyzer.integrate_drt(gamma, tau)

    # calculate and display the error
    eis_err = z_re + 1j * z_im - eis_reg
    rms_err_eis = np.sqrt(np.average(eis_weights * eis_err * np.conj(eis_err))
                          / np.average(eis_weights * (z_re ** 2 + z_im ** 2))).real
    # https://pyformat.info/
    print_to_gui('{:<60.60}'.format(path.split(input_filename)[1]) + ' - Weighted EIS fit RMS error: ' +
                 '{:>6.2f}'.format(rms_err_eis * 100) + '%')

    if plot or save_plot:
        # plot the results
        if eis_draw_lines:
            eis_format_ids = ((1, 0, 0), (2, 8, 0), (2, 10, 1))
        else:
            eis_format_ids = ((1, 0, 0), (1, 8, 2), (1, 10, 3))
        drt_format_ids = ((2, 0, 0),)
        eis_labels = ['Measured', 'DRT Fitted', 'DRT Fitted, L removed']
        if len(filename_no_path) > 30:
            # https://pyformat.info/
            short_filename = '{:.12}'.format(filename_no_path) + ' ... ' + '{:.13}'.format(filename_no_path[::-1])[::-1]
        else:
            short_filename = filename_no_path
        if not batch:
            print_to_gui('Press \'Esc\' to close the plot')
            if hasattr(L, 'len') or hasattr(L, 'size'):
                L_str = '{:.5f}'.format(L[0] * 1e6) + ', ' + '{:.5f}'.format(L[1] * 1e6)
            else:
                L_str = '{:.5f}'.format(L * 1e6)
            print_to_gui("\nL = " + L_str + " uH, R_inf_ref = " + '{:.5f}'.format(R_inf)
                         + " Ohm, R_0 = " + '{:.5f}'.format(R_0) + " Ohm.\n")
            print_to_gui('You can save the plots by clicking \'save\' button on the plot')
        else:
            print_to_gui('Press \'F5\' to redraw the plot, \'Enter\' to plot the next file, \'Esc\' to close the plot.')
        ecm = util_ecm.ECM()
        ecm.peak_damp_dec = peak_damp_dec
        drt_peak_indices = ecm.find_gamma_peaks(gamma, tau)
        hdl = util_plot.plot_eisdrtres([freq, freq, freq], [z_re, eis_reg.real, eis_noL.real],
                                       [z_im, eis_reg.imag, eis_noL.imag],
                                       gamma, tau, drt_L=L, drt_R_inf=R_inf, drt_R_0=R_0,
                                       eis_res_real=z_res_ratio_re,
                                       eis_res_imag=z_res_ratio_im, eis_weights=eis_weights,
                                       eis_title='EIS of ' + short_filename, eis_labels=eis_labels,
                                       eis_format_ids=eis_format_ids, eis_highlight=not eis_draw_lines,
                                       drt_show_LR=drt_show_LR, drt_highlights=drt_peak_indices,
                                       drt_title='DRT of ' + short_filename, drt_format_ids=drt_format_ids,
                                       hdl=hdl, drt_show_tau=drt_show_tau)
        if plot:
            if not connected_to_gui:
                hdl = util_plot.show_plot(hdl)
        if save_plot:
            # prepare output path
            if save_plot_outpath is None:
                save_plot_outpath = path.split(input_filename)[0]
            if path.exists(save_plot_outpath):
                # print_to_gui('Output path already exists!', save_plot_outpath)
                pass
            else:
                makedirs(save_plot_outpath)

            plot_fmt = 'png'
            plot_filename = path.join(save_plot_outpath, filename_no_path)  # change path
            plot_filename = path.splitext(plot_filename)[0] + '_plot.' + plot_fmt
            util_plot.save_plot(hdl, plot_filename, img_fmt=plot_fmt)
        # system('pause')

    if savedata:
        if save_residuals:
            # prepare output path
            if eis_residual_outpath is None:
                eis_residual_outpath = path.split(input_filename)[0]
            if path.exists(eis_residual_outpath):
                # print_to_gui('Output path already exists!', eis_residual_outpath)
                pass
            else:
                makedirs(eis_residual_outpath)

            # save the results
            eis_residual_name = path.join(eis_residual_outpath, filename_no_path)  # change path
            util_io.save_eis_data(path.splitext(eis_residual_name)[0] + '_eis_residual.' + save_fmt,
                                  freq, z_res_ratio_re, z_res_ratio_im, delim=delim)
        # save weights on input EIS data for DRT and ECM evaluation
        if save_weights:
            # prepare output path
            if eis_weights_outpath is None:
                eis_weights_outpath = path.split(input_filename)[0]
            if path.exists(eis_weights_outpath):
                # print_to_gui('Output path already exists!', eis_weights_outpath)
                pass
            else:
                makedirs(eis_weights_outpath)

            # save the results
            eis_weights_name = path.join(eis_weights_outpath, filename_no_path)  # change path
            util_io.save_eis_data(path.splitext(eis_weights_name)[0] + '_eis_weights.' + save_fmt,
                                  freq, eis_weights, eis_weights, delim=delim)
        if save_drt_or_smoothed_eis:
            if kk_test:
                # prepare output path
                if eis_smooth_outpath is None:
                    eis_smooth_outpath = path.split(input_filename)[0]
                if path.exists(eis_smooth_outpath):
                    # print_to_gui('Output path already exists!', eis_smooth_outpath)
                    pass
                else:
                    makedirs(eis_smooth_outpath)

                # save the results
                eis_smooth_name = path.join(eis_smooth_outpath, filename_no_path)  # change path
                util_io.save_eis_data(path.splitext(eis_smooth_name)[0] + '_eis_smooth.' + save_fmt,
                                      freq, eis_noL.real, eis_noL.imag, delim=delim)
            else:
                # prepare output path
                if drt_outpath is None:
                    drt_outpath = path.split(input_filename)[0]
                if path.exists(drt_outpath):
                    # print_to_gui('Output path already exists!', save_filename)
                    pass
                else:
                    makedirs(drt_outpath)

                # save the results
                drt_name = path.join(drt_outpath, filename_no_path)  # change path
                util_io.save_drt_data(path.splitext(drt_name)[0] + '_drt.' + save_fmt,
                                      L, R_inf, gamma, tau, R_0=R_0, delim=delim, write_tau=drt_show_tau)
            if not batch:
                print_to_gui('DRT result or smoothed EIS data saved!')
    hdl.plot_ranges.save(plot_ranges_file)
    # system('pause')
    return hdl


def process_one_file_ecm(input_filename, batch=False, savedata=False, area=1.0, eis_convert_outpath=None,
                         ecm_outpath=None, eis_outpath=None, drtfit_outpath=None, drt_outpath=None,
                         hdl=None, settings=None, save_plot_outpath=None, eis_residual_outpath=None,
                         eis_weights_outpath=None, R_inf_set=None, L_set=None):
    if settings is None:
        settings = Settings()

    max_num_peak = settings.max_num_peak
    eis_fit = settings.eis_fit
    eis_fit_tau = settings.eis_fit_tau
    eis_fit_refresh_alpha = settings.eis_fit_refresh_alpha
    eis_fit_elementwise = settings.eis_fit_elementwise
    eis_draw_lines = settings.eis_draw_lines
    cdrt_fit_coupled = settings.cdrt_fit_coupled
    cdrt_fit_R = settings.cdrt_fit_R
    cdrt_fit_tau = settings.cdrt_fit_tau
    cdrt_fit_alpha = settings.cdrt_fit_alpha
    cdrt_fit_refresh_from_new = settings.cdrt_fit_refresh_from_new
    auto_peak_detect = settings.auto_peak_detect
    drt_show_tau = settings.drt_show_tau
    save_ecm_eis_drt = settings.save_ecm_eis_drt and savedata
    save_ref_drt = settings.save_ref_drt and savedata
    save_weights = settings.save_weights and savedata
    save_fmt = settings.save_fmt
    silent = settings.silent
    plot = settings.show_plot
    save_plot = settings.save_plot and savedata
    save_residuals = settings.save_residuals and savedata
    convert_ism = settings.convert_ism and savedata
    eis_lo_f_discard_decades = settings.eis_lo_f_discard_decades
    eis_hi_f_discard_decades = settings.eis_hi_f_discard_decades
    peak_damp_dec = settings.peak_damp_dec
    gerischer_ids = util_ecm.gerischer_pos_to_id(settings.gerischer_positions)

    kk_test_backup = settings.kk_test
    settings.kk_test = False
    drt_settings = get_drt_settings(settings)
    settings.kk_test = kk_test_backup
    if hdl.data is None:
        ecm = util_ecm.ECM(max_num_peak)
    else:
        ecm = hdl.data
        if ecm.n_rq != max_num_peak:
            ecm = util_ecm.ECM(max_num_peak)
            hdl.hold = False  # Force refit if max_num_peak is changed
    ecm.drt_settings = drt_settings
    ecm.peak_damp_dec = peak_damp_dec

    ret_data = load_eis_and_trim(input_filename, area, batch, save_fmt, convert_ism, eis_convert_outpath,
                                 eis_lo_f_discard_decades, eis_hi_f_discard_decades)
    eis_data, (filename_no_path, ext, delim) = ret_data
    if eis_data is None:
        return hdl
    if batch:
        print_to_gui('\n--------------------\nProcessing ' + filename_no_path + ' ...')

    # fit with ECM
    freq = eis_data[:, 0]
    if eis_data.shape[1] > 3:
        # elapsed_time = eis_data[:, 3]
        eis_significance = eis_data[:, 4]
    else:
        eis_significance = np.ones(len(freq))
    # hdl.hold means keeping the reference data and fit results. If hold, INHERIT previous fit results
    m_ecm_pars = None
    if not auto_peak_detect:
        from EISART_support import manual_ecm_pars
        m_ecm_pars = manual_ecm_pars
    if eis_fit:
        inherit = hdl.hold and (not eis_fit_refresh_alpha)
        ecm.fit_eis(freq, eis_data[:, 1], eis_data[:, 2], inherit=inherit, fix_tau=(not eis_fit_tau),
                    silent=silent, refine_alpha=False, gerischer_ids=gerischer_ids,
                    eis_significance=eis_significance, fit_elementwise=eis_fit_elementwise,
                    m_ecm_pars=m_ecm_pars, R_inf_set=R_inf_set, L_set=L_set)
    else:
        inherit = hdl.hold and (not cdrt_fit_refresh_from_new)
        ecm.fit_cdrt(freq, eis_data[:, 1], eis_data[:, 2], silent=silent, inherit=inherit,
                     fit_coupled=cdrt_fit_coupled,
                     fit_R=cdrt_fit_R, fit_tau=cdrt_fit_tau, fit_alpha=cdrt_fit_alpha, gerischer_ids=gerischer_ids,
                     eis_significance=eis_significance, m_ecm_pars=m_ecm_pars, R_inf_set=R_inf_set, L_set=L_set)
    hdl.data = ecm

    # save reference DRT
    if save_ref_drt:
        if drt_outpath is None:
            drt_outpath = path.split(input_filename)[0]
        if path.exists(drt_outpath):
            # print_to_gui('Output path already exists!', save_filename)
            pass
        else:
            makedirs(drt_outpath)
        drt_name = path.join(drt_outpath, filename_no_path)  # change path
        R_0 = ecm.R_inf_ref + ecm.cdrt_ref[-1] - ecm.cdrt_ref[0]
        util_io.save_drt_data(path.splitext(drt_name)[0] + '_drt.' + save_fmt,
                              ecm.L_ref, ecm.R_inf_ref, ecm.gamma_ref, ecm.tau_out,
                              R_0=R_0, delim=delim, write_tau=drt_show_tau)

    # prepare EIS and DRT data
    eis_ecm = ecm.eis_gen(freq)
    drt_analyzer = util_tikhonov.DRTAnalyzer()
    drt_analyzer.settings = ecm.drt_settings
    L, R_inf, gamma, tau, _, _ = \
        drt_analyzer.drt_iterative(freq, eis_ecm.real, eis_ecm.imag, eis_significance=eis_significance,
                                   R_inf_set=R_inf_set, L_set=L_set)
    R_0 = R_inf + ecm.cdrt_gen[-1] - ecm.cdrt_gen[0]

    z_abs = np.sqrt(eis_data[:, 1] ** 2 + eis_data[:, 2] ** 2)
    z_res_ratio_re = -(eis_data[:, 1] - eis_ecm.real) / z_abs
    z_res_ratio_im = -(eis_data[:, 2] - eis_ecm.imag) / z_abs

    # save ECM parameters
    if savedata:
        if ecm_outpath is None:
            ecm_outpath = path.split(input_filename)[0]
        if path.exists(ecm_outpath):
            # print_to_gui('Output path already exists!', ecm_outpath)
            pass
        else:
            makedirs(ecm_outpath)
        ext = path.splitext(input_filename)[1]
        save_filename = path.splitext(path.join(ecm_outpath, filename_no_path))[0] + '_ecm.' + save_fmt
        ZView_mdl_filename = path.splitext(path.join(ecm_outpath, filename_no_path))[0] + '_ecm.' + 'mdl'
        print_to_gui('ECM results saved.')
    else:
        save_filename = None
        ZView_mdl_filename = None
    fit_result = util_io.save_ecm(save_filename, ecm)
    mdl_io = util_io.ECM_ZView_IO()
    mdl_io.save_ecm_as_mdl(ZView_mdl_filename, ecm, m_ecm_pars)
    if not silent:
        print_to_gui(fit_result)

        # calculate and display the error
        eis_err = eis_data[:, 1] + 1j * eis_data[:, 2] - eis_ecm
        rms_err_eis = np.sqrt(np.average(ecm.eis_weights * eis_err * np.conj(eis_err))
                              / np.average(ecm.eis_weights * (eis_data[:, 1] ** 2 + eis_data[:, 2] ** 2))).real
        # https://pyformat.info/
        print_to_gui('{:<60.60}'.format(path.split(input_filename)[1]) + ' - Weighted EIS fit RMS error: ' +
                     '{:>6.2f}'.format(rms_err_eis * 100) + '%')

    # plot fitting results
    eis_z_indiv = None
    if plot or save_plot:
        if eis_draw_lines:
            eis_format_ids = ((1, 0, 0), (2, 8, 0))
        else:
            eis_format_ids = ((1, 0, 0), (1, 8, 2))
        drt_format_ids = ((2, 0, 0), (2, 8, 1))
        eis_labels = ('Input', 'ECM')
        drt_labels = eis_labels
        if len(filename_no_path) > 30:
            # https://pyformat.info/
            short_filename = '{:.12}'.format(filename_no_path) + ' ... ' + '{:.13}'.format(filename_no_path[::-1])[::-1]
        else:
            short_filename = filename_no_path

        def find_indices(array, values):
            # find the corresponding index in array (sorted) that is closest to each element in values
            n = len(values)
            indices = np.zeros(n, dtype=int)
            for i in range(n):
                indices[i] = np.argmin(np.abs(array - values[i]))
            return indices

        R_0_ref = ecm.R_inf_ref + ecm.cdrt_ref[-1] - ecm.cdrt_ref[0]
        ecm_peak_indices = find_indices(np.log(ecm.tau_out), np.log(ecm.rq_pars['tau']))
        if not (silent or batch):
            print_to_gui('Press \'F5\' to redraw the plot, \'Enter\' to plot the next file, \'Esc\' to close the plot.')
        eis_z_indiv = ecm.eis_indiv_cummulative(ecm.eis_gen(freq, indiv_elements=True), R_inf=ecm.R_inf_ref)
        hdl = util_plot.plot_eisdrtres([freq, freq], [eis_data[:, 1], eis_ecm.real], [eis_data[:, 2], eis_ecm.imag],
                                       [ecm.gamma_ref, gamma], [ecm.tau_out, tau],
                                       eis_res_real=z_res_ratio_re, eis_res_imag=z_res_ratio_im,
                                       eis_z_indiv=eis_z_indiv, drt_gamma_indiv=ecm.gamma_gen_indiv,
                                       eis_labels=eis_labels, eis_weights=ecm.eis_weights,
                                       eis_title='EIS of ' + short_filename,
                                       eis_format_ids=eis_format_ids, eis_highlight=not eis_draw_lines,
                                       drt_highlights=ecm_peak_indices,
                                       drt_L=[ecm.L_ref, L], drt_R_inf=[ecm.R_inf_ref, R_inf], drt_R_0=[R_0_ref, R_0],
                                       drt_show_LR=False, drt_title='DRT of ' + short_filename, drt_show_tau=drt_show_tau,
                                       drt_format_ids=drt_format_ids, drt_labels=drt_labels, hdl=hdl)
        if plot:
            if not connected_to_gui:
                hdl = util_plot.show_plot(hdl)
        if save_plot:
            # prepare output path
            if save_plot_outpath is None:
                save_plot_outpath = path.split(input_filename)[0]
            if path.exists(save_plot_outpath):
                # print_to_gui('Output path already exists!', save_plot_outpath)
                pass
            else:
                makedirs(save_plot_outpath)

            plot_fmt = 'png'
            plot_filename = path.join(save_plot_outpath, filename_no_path)  # change path
            plot_filename = path.splitext(plot_filename)[0] + '_plot.' + plot_fmt
            util_plot.save_plot(hdl, plot_filename, img_fmt=plot_fmt)

        # if the fit would be done in <2s
        quick_fit = eis_fit or (not eis_fit) and (not cdrt_fit_coupled)
        if quick_fit:
            pass
        else:
            hdl.fig, hdl.axs = None, None  # clear handle because the fit takes too long to wait
            # the above line may be removed if the cdrt fit gets faster (<2s)
            util_plot.close_plot()

    if savedata:
        # save fitting residual
        if save_residuals:
            # prepare output path
            if eis_residual_outpath is None:
                eis_residual_outpath = path.split(input_filename)[0]
            if path.exists(eis_residual_outpath):
                # print_to_gui('Output path already exists!', eis_residual_outpath)
                pass
            else:
                makedirs(eis_residual_outpath)
            eis_residual_name = path.join(eis_residual_outpath, filename_no_path)  # change path
            util_io.save_eis_data(path.splitext(eis_residual_name)[0] + '_eis_residual.' + save_fmt,
                                  freq, z_res_ratio_re, z_res_ratio_im, delim=delim)

        # save weights on input EIS data for DRT and ECM evaluation
        if save_weights:
            # prepare output path
            if eis_weights_outpath is None:
                eis_weights_outpath = path.split(input_filename)[0]
            if path.exists(eis_weights_outpath):
                # print_to_gui('Output path already exists!', eis_weights_outpath)
                pass
            else:
                makedirs(eis_weights_outpath)

            # save the results
            eis_weights_name = path.join(eis_weights_outpath, filename_no_path)  # change path
            util_io.save_eis_data(path.splitext(eis_weights_name)[0] + '_eis_weights.' + save_fmt,
                                  freq, ecm.eis_weights, ecm.eis_weights, delim=delim)

        # save EIS and DRT data of ECM
        if save_ecm_eis_drt:
            # prepare output path
            if drtfit_outpath is None:
                drtfit_outpath = path.split(input_filename)[0]
            if path.exists(drtfit_outpath):
                # print_to_gui('Output path already exists!', drt_outpath)
                pass
            else:
                makedirs(drtfit_outpath)

            if eis_outpath is None:
                eis_outpath = path.split(input_filename)[0]
            if path.exists(eis_outpath):
                # print_to_gui('Output path already exists!', eis_outpath)
                pass
            else:
                makedirs(eis_outpath)

            # save the results
            drt_name = path.join(drtfit_outpath, filename_no_path)  # change path
            eis_fitname = path.join(eis_outpath, filename_no_path)  # change path

            util_io.save_drt_data(path.splitext(drt_name)[0] + '_drt_ecm.' + save_fmt,
                                  L, R_inf, gamma, tau, R_0=R_0, delim=delim, write_tau=drt_show_tau)
            if eis_z_indiv is None:
                eis_z_indiv = ecm.eis_indiv_cummulative(ecm.eis_gen(freq, indiv_elements=True), R_inf=ecm.R_inf_ref)
            util_io.save_ecm_indiv_eis_data(path.splitext(eis_fitname)[0] + '_eis_ecm.' + save_fmt,
                                            freq, eis_ecm, eis_z_indiv, delim=delim)
            if not silent:
                print_to_gui('EIS from the fitted ECM and its DRT saved!')
    hdl.plot_ranges.save(plot_ranges_file)
    return hdl


class Settings:
    def __init__(self, filename=None):
        self.convert_ism = True
        self.save_ecm_eis_drt = True
        self.save_plot = True  # If True, save plots as *.png files
        self.save_residuals = True  # if True, save fitting residuals
        self.save_ref_drt = True
        self.save_weights = True
        self.save_fmt = 'txt'

        self.show_plot = True
        self.silent = False
        self.drt_show_tau = False
        self.drt_show_LR = True
        self.eis_draw_lines = True

        self.drt_not_ecm = False  # if True, plot DRT or smoothed EIS with residuals, else plot ECM fitting results

        self.kk_test = False
        self.drt_false_peak_suppress = True
        self.auto_lmd = False

        self.eis_fit = True
        self.eis_fit_tau = True
        self.eis_fit_refresh_alpha = True
        self.eis_fit_elementwise = False
        self.cdrt_fit_coupled = False
        self.cdrt_fit_R = True
        self.cdrt_fit_tau = True
        self.cdrt_fit_alpha = True
        self.cdrt_fit_refresh_from_new = True
        self.auto_peak_detect = True

        self.lmd = 1e-3
        self.eis_lo_f_discard_decades = 0.0
        self.eis_hi_f_discard_decades = 0.0
        self.drt_lo_f_extension_decades = 0.0
        self.drt_hi_f_extension_decades = 0.0
        # self.hyper_lmd_iter = 0
        self.auto_lmd_radius_dec = 1.0
        self.weight_iter = 0

        self.max_num_peak = 7
        self.peak_damp_dec = 0.1
        self.max_trim = 10
        self.flt_level = 0.8
        self.trim_unskip = 5
        self.gerischer_positions = []

        if filename is not None:
            self.load(filename)

    def load(self, filename):
        if not path.exists(filename):
            # raise ValueError("Plot settings file \'" + filename + '\' does not exist.')
            self.save(filename)
        else:
            with open(filename, 'r') as f:
                s = f.read()
                s = s.replace(',\n\'', ', \'')
                self.from_str(s)

    def save(self, filename):
        def save_file():
            with open(filename, 'w+') as f:
                s = str(self)
                s = s.replace(', \'', ',\n\'')
                f.write(s)  # TODO may overwrite without warning
        try:
            save_file()
        except PermissionError:
            from time import sleep
            sleep(0.5)
            save_file()

    def from_str(self, s):
        dict_ = eval(s)  # evaluate the string. The result is expected to be a dictionary
        for key, val in dict_.items():
            setattr(self, key, val)
        return self

    def __str__(self):
        return str(self.__dict__)  # convert this instance to a dictionary, then to a string


settings = Settings()


def load_settings():
    global settings
    settings = Settings(settings_file)


def kernel_main(area=1.0, head='', save_results=False, R_inf_set=None, L_set=None):
    global settings
    try:
        load_settings()
        if connected_to_gui:
            if len(head) == 0:
                return 1
        else:
            while len(head) == 0:
                # get path to EIS data file
                head = input('\nPlease drop an EIS data file or folder here.\n>> ')
        # compatibility for path names with whitespace or special chars: remove quotation marks around path (head)
        if head[0] == '\"' and head[-1] == '\"':
            head = head[1:-1]

        if len(path.splitext(head)[1]) > 0:
            abort = True
            if path.splitext(head)[1][1:] in util_io.supported_extension:
                try:
                    data = util_io.load_eis_data(head)  # here data is read and discarded
                    if data.size > 0:
                        abort = False
                except ValueError:
                    pass
            if abort:
                print_to_gui('Unsupported data format!\n')
                if connected_to_gui:
                    from EISART_support import message_set
                    message_set('Unsupported data format!')
                return 1

        ecm_foldername = 'ECM'
        drtfit_foldername = 'DRT_ECM'
        eisfit_foldername = 'EIS_ECM'
        drt_foldername = 'DRT'
        eis_smooth_foldername = 'EIS_Smooth'
        eis_convert_foldername = 'EIS_from_ism'
        save_plot_foldername = 'EISART_Plots'
        eis_residual_foldername = 'EIS_Residual'
        eis_weights_foldername = 'EIS_Weights'

        if not connected_to_gui:
            area = 0
            while area <= 0:
                try:
                    area = float(input('Please specify the active electrode area in cm^2\n>> '))
                except ValueError:
                    print_to_gui('Please specify with a positive number, i.e. 1.0 or 80 or 4.5e-2.')
                    continue
            # response = input('Input anything to plot DRT residuals instead of ECM. '
            #                  'To fit with ECM just press Enter\n>> ')
            # drt_not_ecm = (len(response) > 0)
            response = input('Input anything to save the results. Otherwise just press Enter.\n>> ')
            save_results = (len(response) > 0)

        hdl = util_plot.Handle()
        # hdl.hold is bool. If True, process the same file again
        # hdl.data is an instance of ECM (equivalent circuit model)
        if connected_to_gui:
            from EISART_support import fig_gui
            hdl.fig = fig_gui
            hdl.plot_ranges = util_plot.PlotRanges(filename=plot_ranges_file)
        if path.isfile(head):
            print_to_gui('Processing ' + head)
            directory = path.split(head)[0]
            ecm_outpath = path.join(directory, ecm_foldername)
            drtfit_outpath = path.join(directory, drtfit_foldername)
            eisfit_outpath = path.join(directory, eisfit_foldername)
            drt_outpath = path.join(directory, drt_foldername)
            eis_smooth_outpath = path.join(directory, eis_smooth_foldername)
            eis_convert_outpath = path.join(directory, eis_convert_foldername)
            save_plot_outpath = path.join(directory, save_plot_foldername)
            eis_residual_outpath = path.join(directory, eis_residual_foldername)
            eis_weights_outpath = path.join(directory, eis_weights_foldername)
            while True:
                load_settings()
                drt_not_ecm = settings.drt_not_ecm
                if drt_not_ecm:
                    hdl = process_one_file_drt(head, batch=False, savedata=save_results, area=area,
                                               drt_outpath=drt_outpath, eis_smooth_outpath=eis_smooth_outpath,
                                               eis_convert_outpath=eis_convert_outpath,
                                               save_plot_outpath=save_plot_outpath,
                                               hdl=hdl, settings=settings, eis_weights_outpath=eis_weights_outpath,
                                               eis_residual_outpath=eis_residual_outpath,
                                               R_inf_set=R_inf_set, L_set=L_set)
                else:
                    hdl = process_one_file_ecm(head, batch=False, area=area, ecm_outpath=ecm_outpath,
                                               eis_convert_outpath=eis_convert_outpath,
                                               eis_outpath=eisfit_outpath, drtfit_outpath=drtfit_outpath, hdl=hdl,
                                               savedata=save_results, settings=settings, drt_outpath=drt_outpath,
                                               save_plot_outpath=save_plot_outpath,
                                               eis_weights_outpath=eis_weights_outpath,
                                               eis_residual_outpath=eis_residual_outpath,
                                               R_inf_set=R_inf_set, L_set=L_set)
                if not hdl.hold:
                    hdl.data = None  # initialize ECM
                    break
            if not connected_to_gui:
                util_plot.close_plot()
        elif path.isdir(head):
            print_to_gui('Processing supported files in ' + head)
            nameList = listdir(head)
            nameList = [path.join(head, filename) for filename in nameList]
            ecm_outpath = path.join(head, ecm_foldername)
            drtfit_outpath = path.join(head, drtfit_foldername)
            eisfit_outpath = path.join(head, eisfit_foldername)
            drt_outpath = path.join(head, drt_foldername)
            eis_smooth_outpath = path.join(head, eis_smooth_foldername)
            eis_convert_outpath = path.join(head, eis_convert_foldername)
            save_plot_outpath = path.join(head, save_plot_foldername)
            eis_residual_outpath = path.join(head, eis_residual_foldername)
            eis_weights_outpath = path.join(head, eis_weights_foldername)
            for full_filename in nameList:
                hdl.hold = False  # hdl.hold is bool. If True, process the same file again
                while True:
                    load_settings()
                    drt_not_ecm = settings.drt_not_ecm
                    if drt_not_ecm:
                        hdl = process_one_file_drt(full_filename, batch=True, savedata=save_results, area=area,
                                                   drt_outpath=drt_outpath, eis_smooth_outpath=eis_smooth_outpath,
                                                   eis_convert_outpath=eis_convert_outpath,
                                                   save_plot_outpath=save_plot_outpath,
                                                   hdl=hdl, settings=settings, eis_weights_outpath=eis_weights_outpath,
                                                   eis_residual_outpath=eis_residual_outpath,
                                                   R_inf_set=R_inf_set, L_set=L_set)
                    else:
                        hdl = process_one_file_ecm(full_filename, batch=True, area=area, ecm_outpath=ecm_outpath,
                                                   eis_outpath=eisfit_outpath, drtfit_outpath=drtfit_outpath,
                                                   hdl=hdl, save_plot_outpath=save_plot_outpath,
                                                   eis_convert_outpath=eis_convert_outpath,
                                                   savedata=save_results, settings=settings,
                                                   drt_outpath=drt_outpath, eis_weights_outpath=eis_weights_outpath,
                                                   eis_residual_outpath=eis_residual_outpath,
                                                   R_inf_set=R_inf_set, L_set=L_set)
                    if not hdl.hold:
                        hdl.data = None  # initialize ECM
                        break
                if connected_to_gui and settings.show_plot:
                    from EISART_support import refresh_gui
                    refresh_gui()
            if not connected_to_gui:
                util_plot.close_plot()
            if save_results:
                print_to_gui('Results saved.')
        else:
            print_to_gui("Input data unspecified! Please input the path to an EIS data file or folder")
        # system('pause')
        print_to_gui('\n--------------------\n')
        return 0
    except Exception:
        # references:
        # https://docs.python.org/3/library/traceback.html
        # https://docs.python.org/3/library/io.html
        import io
        logger = io.StringIO()
        traceback.print_exc(file=logger)
        traceback_str = logger.getvalue()
        print_to_gui(traceback_str, file=sys.stderr)
        print_to_gui('Unexpected error as shown in traceback')


if __name__ == '__main__':
    try:
        print('\n>>>   >>>   >>>   >>>   EISART   <<<   <<<   <<<   <<<\n'
              'Electrochemical Impedance Spectra Analysis & Refining Tool\n'
              'code by Hangyue Li, THU\n')
        # analyze data
        head = ''
        if len(sys.argv) > 1:
            head = sys.argv[1]
        while True:
            exit_code = kernel_main(head=head)
            head = ''
    except Exception:
        traceback.print_exc()
        print('Unexpected error as shown in traceback')
        if not connected_to_gui:
            util_plot.close_plot()
        system('pause')
