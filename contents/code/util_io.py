"""
File read & write interface
Code by Hangyue Li, Tsinghua University
"""

import numpy as np
from os import path
from calendar import timegm
from time import strptime


supported_extension = ('txt', 'csv', 'ism')


def load_eis_data(filename, read_comments=False, title_delim='\t'):
    comments = ''
    filename = filename.replace('\'', '').replace('\"', '').replace('&', '').strip(' ')
    ext = path.splitext(filename)[1][1:]  # remove the point: '.txt' -> 'txt'
    time_fmt = '%Y/%m/%d_%H:%M:%S'

    def _read_date_time(date_time_str):
        try:
            # expected time format: '2021/01/02_03:04:05.678'
            time_bulk_str, time_frac_str = date_time_str.split('.')
            _epoch_time = timegm(strptime(time_bulk_str, time_fmt)) + float('0.' + time_frac_str)
        except ValueError:
            _epoch_time = 0.0
        return _epoch_time

    def _read_text_row(row_value_list):
        if len(row_value_list) == 5:  # assert columns: freq, z_re, z_im, date_time, significance
            try:
                time_val = float(row_value_list[3])
            except ValueError:  # date-time
                time_val = _read_date_time(row_value_list[3])
            try:
                signif = float(row_value_list[4])  # significance (a quality measure) of eis data
            except ValueError:
                signif = 1.0
            _eis_row = [float(row_value_list[0]), float(row_value_list[1]), float(row_value_list[2]), time_val, signif]
        else:
            _eis_row = []
            for row_value in row_value_list:
                try:
                    _eis_row.append(float(row_value))
                except ValueError:
                    pass
        return _eis_row

    def _regularize_eis_data(raw_eis_data):
        res_eis_data = []
        for row in raw_eis_data:
            # remove all rows with less than 3 values
            # resulted from rows with the proper count of values
            #     but some of the values cannot be converted to floats (i.e. the title line)
            if len(row) >= 3:
                res_eis_data.append(row)
        row_len_aver = np.average([len(row) for row in res_eis_data])
        for i, row in enumerate(res_eis_data):
            if abs(len(row) - row_len_aver) > 0.1:
                del res_eis_data[i]
        row_len_var = np.var([len(row) for row in res_eis_data])
        if row_len_var > 1e-12:  # if the length of the rows varies
            res_eis_data = [row[:3] for row in res_eis_data]  # discard the 4th and 5th column
        res_eis_data = np.array(res_eis_data)
        if len(res_eis_data[0]) == 5:  # file converted from ism with 'Zahner Binary File Converter' by Hangyue Li
            time_seq = res_eis_data.T[3]
            time_not_ascending = np.min(time_seq[1:] - time_seq[:-1]) < 0
            signif_seq = res_eis_data.T[4]
            signif_out_of_range = np.any(signif_seq > 1) or np.any(signif_seq < 0)
            if time_not_ascending or signif_out_of_range:
                res_eis_data = res_eis_data[:, :3]
        elif len(res_eis_data[0]) == 4 or len(res_eis_data[0]) == 6:
            # might be file converted from ism with Zahner software
            first_column = res_eis_data.T[0]
            if np.abs(np.average(first_column[1:] - first_column[:-1] - 1)) < 1e-6:
                # then the first column is a list of data point number like 1, 2, 3, 4, 5, ...
                res_eis_data = res_eis_data[:, 1:4]  # discard the number column
        return res_eis_data

    if ext == supported_extension[0]:  # txt
        with open(filename, 'r') as datafile:
            text = datafile.read()
        while '\t\t' in text:
            text = text.replace('\t\t', ' ')
        while '  ' in text:
            text = text.replace('  ', ' ')
        text = text.replace(' ', '\t')
        # eis_data = np.loadtxt(filename, dtype=np.float, delimiter='\t')
        # eis_data = np.fromstring(text, dtype=np.float, sep='\t').reshape(-1, 3)
        eis_data = []
        text_rows = text.split('\n')
        for text_row in text_rows:
            value_list = text_row.split('\t')
            if len(value_list) < 3:
                continue  # skip this row if it contains less than 3 values
            try:
                eis_row = _read_text_row(value_list)
                eis_data.append(eis_row)
            except ValueError:
                continue  # skip this row if ValueError is raised when parsing the entries as floats
        eis_data = _regularize_eis_data(eis_data)
    elif ext == supported_extension[1]:  # csv
        # eis_data = np.loadtxt(filename, dtype=np.float, delimiter=',')
        with open(filename, 'r') as datafile:
            text = datafile.read()
        eis_data = []
        text_rows = text.split('\n')
        for text_row in text_rows:
            value_list = text_row.split(',')
            if len(value_list) < 3:
                continue  # skip this row if it contains less than 3 values
            try:
                eis_row = _read_text_row(value_list)
                eis_data.append(eis_row)
            except ValueError:
                continue  # skip this row if ValueError is raised when parsing the entries as floats
        eis_data = _regularize_eis_data(eis_data)
    elif ext == supported_extension[2]:  # ism
        with open(filename, 'rb') as datafile:
            buff = datafile.read()

        # To understand the binary data from scratch, use diskgenius or any other apps to show the data as hex string
        # Try converting the hex to doubles or floats: https://gregstoll.com/~gregstoll/floattohex/

        # format_identifier = buff[0:  8]  # b'\x00\x00\xff\xff\xff\xf0\x00\x00'
        freq_count = buff[8: 12]
        freq_count = int.from_bytes(freq_count, byteorder='big', signed=True) + 1
        float_buff = np.zeros(freq_count)
        curser = 12
        float_bytes = 8  # double precision floats
        dtype_str = '>f' + str(float_bytes)

        for i in range(freq_count):
            byte_section = buff[curser + float_bytes * i: curser + float_bytes * (i + 1)]
            float_buff[i] = float(np.frombuffer(byte_section, dtype=dtype_str)[0])
        freqs = float_buff * 1
        curser += float_bytes * freq_count

        for i in range(freq_count):
            byte_section = buff[curser + float_bytes * i: curser + float_bytes * (i + 1)]
            float_buff[i] = float(np.frombuffer(byte_section, dtype=dtype_str)[0])
        z_mag = float_buff * 1
        curser += float_bytes * freq_count

        for i in range(freq_count):
            byte_section = buff[curser + float_bytes * i: curser + float_bytes * (i + 1)]
            float_buff[i] = float(np.frombuffer(byte_section, dtype=dtype_str)[0])
        z_phs = float_buff * 1  # radius
        curser += float_bytes * freq_count

        # datetime of each frequency
        for i in range(freq_count):
            byte_section = buff[curser + float_bytes * i: curser + float_bytes * (i + 1)]
            epoch_time = np.frombuffer(byte_section, dtype=dtype_str)[0]
            # Try to understand the large floating point numbers: https://www.epochconverter.com/
            # https://stackoverflow.com/questions/3694487/in-python-how-do-you-convert-seconds-since-epoch-to-a-datetime-object
            float_buff[i] = epoch_time
        # Zahner epoch time is 10 years less than the standard epoch time starting 1970
        time_fmt = '%Y/%m/%d'
        time_difference = timegm(strptime('1980/01/01', time_fmt))
        epoch_time = float_buff + time_difference
        curser += float_bytes * freq_count

        int_bytes = 2
        # significance of each frequency (data quality)
        for i in range(freq_count):
            byte_section = buff[curser + int_bytes * i: curser + int_bytes * (i + 1)]
            permille = int.from_bytes(byte_section, byteorder='big')
            float_buff[i] = permille / 1000.0
        significance = float_buff * 1

        z_re = z_mag * np.cos(z_phs)
        z_im = z_mag * np.sin(z_phs)
        curser += int_bytes * freq_count
        eis_data = np.zeros([freq_count, 5])
        eis_data[:, 0] = freqs
        eis_data[:, 1] = z_re
        eis_data[:, 2] = z_im
        eis_data[:, 3] = epoch_time
        eis_data[:, 4] = significance
        # the rest of the ism file are comments
        comments = buff[curser:]
    else:
        raise ValueError('Unsupported data format!')
    if eis_data.size == 0:
        raise ValueError('Zero-sized eis_data loaded.')
    if read_comments:
        comments = str(comments)[2:-1]  # str(bytes) -> b'xxxxxx'
        comments = comments.swapcase()
        comments = comments.replace('\\X00\\X03', '\nArea: ')
        comments = comments.replace('\\X00\\X06', 'Date(DDMMYY): ')
        comments = comments.replace('\\X00\\X04', '\n')
        comments = comments.replace('\\X00\\X05', '\n')
        comments = comments.replace('\\X00\\X07', '\n')
        comments = comments.replace('\\X00\\X08', '\n')
        comments = comments.replace('\\X00\\X09', '\n')
        comments = comments.replace('\\X00\\X0A', '\n')  # '\x0a' == '\n'
        comments = comments.replace('\\X00\\N', '\n')
        comments = comments.replace('\\X00\\R', '\n')
        comments = comments.replace('\\X00\\X0B', '\n')
        comments = comments.replace('\\X00\\T', '\n')
        comments = comments.replace('\\X00\\X0C', '\n')
        comments = comments.replace('\\X00\\X0D', '\n')
        comments = comments.replace('\\X00\\X0E', '\n')
        comments = comments.replace('\\X00\\X0F', '\n')
        comments = comments.replace('\\X00\\X10', '\n')
        comments = comments.replace('\\X00\\X11', '\n')
        comments = comments.replace('\\X00\\X12', '\n')
        comments = comments.replace('\\X00\\X13', '\n')
        comments = comments.replace('\\X00\\X14', '\n')
        comments = comments.replace('\\X00\\X15', '\n')
        comments = comments.replace('\\X00\\X16', '\n')
        comments = comments.replace('\\X00\\X00', '')
        comments_end = comments.find('\\X01\\X00')
        if comments_end >= 0:
            comments = comments[comments.find('Date'): comments_end]  # discard unrecognized data
        else:
            comments = comments[comments.find('Date'):]  # discard unrecognized data
        comments = comments + '\n\n'
        titles = 'Frequency(Hz)' + title_delim + 'Z_real(Ohm)' + title_delim + 'Z_imag(Ohm)' \
                 + title_delim + 'Time(s)' + title_delim + 'Significance'
        comments = comments + titles
        return eis_data, comments
    else:
        return eis_data


def save_drt_data(filename, L, R_inf, gamma, tau, R_0=None, add_LR=True, delim=',', write_tau=True):
    gamma *= (gamma > 1e-300)  # suppress very small values to zeros
    if not hasattr(L, '__len__'):
        L = np.array([L, 0])
    # data_to_save = np.vstack(L, [np.hstack([R_inf_ref, 0]), np.vstack([gamma, tau]).T])
    header = ''
    if add_LR:
        header = 'L_self (uH)' + delim + '{:.8e}'.format(L[0] * 1e6) \
                 + '\nL_wire (equiv. uH)' + delim + '{:.8e}'.format(L[1] * 1e6)\
                 + '\nR_inf_ref (Ohm * cm^2)' + delim + '{:.8e}'.format(R_inf)
        if R_0 is not None:
            header += '\nR_0 (Ohm * cm^2)' + delim + '{:.8e}'.format(R_0)
    header += '\n'
    if write_tau:
        header += 'tau (s)' + delim + 'gamma'
        data_to_save = np.vstack([tau, gamma]).T
    else:
        header += 'freq (Hz)' + delim + 'gamma'
        data_to_save = np.vstack([1 / (2 * np.pi * tau), gamma]).T
    np.savetxt(filename, data_to_save, delimiter=delim, fmt='%.8e', header=header, comments='')


def save_eis_data(filename, freq, z_re, z_im, delim=','):
    eis_mat = np.vstack([freq, z_re, z_im]).T
    np.savetxt(filename, eis_mat, delimiter=delim, fmt='%.8e')


def save_ecm_indiv_eis_data(filename, freq, z_total, z_indiv=None, delim=','):
    if z_indiv is None:
        save_eis_data(filename, freq, z_total.real, z_total.imag, delim)  # no header, 3 columns: freq, z_re, z_im
    else:
        try:
            vectors = [freq, z_total.real, z_total.imag]
            for z in z_indiv:
                vectors.append(z.real)
                vectors.append(z.imag)
            eis_mat = np.vstack(vectors).T
            header = 'Frequency (Hz)' + delim + 'Z_total_re (Ohm*cm^2)' + delim + 'Z_total_im'
            for i in range(len(z_indiv)):
                z_name = 'Z_' + str(i + 1)
                header += delim + z_name + '_re' + delim + z_name + '_im'
            np.savetxt(filename, eis_mat, delimiter=delim, fmt='%.8e', header=header)
        except (ValueError, TypeError):
            # as if z_indiv is None
            save_eis_data(filename, freq, z_total.real, z_total.imag, delim)  # no header, 3 columns: freq, z_re, z_im


def save_ecm(filename, ecm):
    string = ecm.save(filename)
    if filename is not None:
        with open(filename, 'w+') as ecm_file:
            ecm_file.write(string)
    return string


def read_isw(isw_filename, with_comments=False, title_delim='\t'):
    ext = path.splitext(isw_filename)[1][1:]  # remove the point: '.isw' -> 'isw'
    if ext != 'isw':
        return None
    with open(isw_filename, 'rb') as f:
        byte_array = f.read()
    pointer = 0
    # format_identifier = byte_array[0: 8]  # b'\x00\x00\xff\xff\xff\xfe\x00\x00'
    # unknown_data01    = byte_array[8:16]  # b'\x00\x00\x00\x0e\x00\x00\x00\x00'
    n_data_points = int.from_bytes(byte_array[16:18], byteorder='big', signed=False)
    pointer += 18
    # read data section
    float_bytes = 8
    n_columns = 3
    dtype_str = '>f' + str(float_bytes)
    float_buff = np.zeros([n_data_points, n_columns])
    for i in range(n_data_points):
        for j in range(n_columns):
            byte_section = byte_array[pointer: pointer + float_bytes]
            float_buff[i, j] = np.frombuffer(byte_section, dtype=dtype_str)[0] / 1e3
            pointer += float_bytes
    comments = byte_array[pointer + 28:]
    if with_comments:
        comments = str(comments)[2:-1]
        comments = comments.swapcase()
        comments = comments.replace('\\X00\\X03', '\n')
        comments = comments.replace('\\X00\\X06', '\n')
        comments = comments.replace('\\X00\\X04', '\n')
        comments = comments.replace('\\X00\\X05', '\n')
        comments = comments.replace('\\X00\\X07', '\n')
        comments = comments.replace('\\X00\\X08', '\n')
        comments = comments.replace('\\X00\\X09', '\n')
        comments = comments.replace('\\X00\\X0A', '\n')  # '\x0a' == '\n'
        comments = comments.replace('\\X00\\N', '\n')
        comments = comments.replace('\\X00\\X0B', '\n')
        comments = comments.replace('\\X00\\T', '\n')
        comments = comments.replace('\\X00\\X0C', '\n')
        comments = comments.replace('\\X00\\X0D', '\n')
        comments = comments.replace('\\X00\\X0E', '\n')
        comments = comments.replace('\\X00\\X0F', '\n')
        comments = comments.replace('\\X00\\X10', '\n')
        comments = comments.replace('\\X00\\X11', '\n')
        comments = comments.replace('\\X00\\X12', '\n')
        comments = comments.replace('\\X00\\X13', '\n')
        comments = comments.replace('\\X00\\X14', '\n')
        comments = comments.replace('\\X00\\X15', '\n')
        comments = comments.replace('\\X00\\X16', '\n')
        comments = comments.replace('\\X00\\X00', '')
        comments_end = comments.find('\\XFF\\XFF')
        if comments_end >= 0:
            comments = comments[: comments_end]  # discard unrecognized data
        else:
            comments = comments[:]  # discard unrecognized data
        comments += '\n\n'
        if comments.startswith('\n'):
            comments = comments[1:]
        titles = 'Voltage(V)' + title_delim + 'Current(A)' + title_delim + 'Time(s)'
        comments = comments + titles
        return float_buff, comments
    else:
        return float_buff


def read_isp(isw_filename, with_comments=False, title_delim='\t'):
    ext = path.splitext(isw_filename)[1][1:]  # remove the point: '.isp' -> 'isp'
    if ext != 'isp':
        return None
    with open(isw_filename, 'rb') as f:
        byte_array = f.read()
    pointer = 0
    # format_identifier = byte_array[0: 8]  # b'\x00\x00\xff\xff\xff\xff\x00\x00'
    n_data_points = int.from_bytes(byte_array[8:12], byteorder='big', signed=False) + 1
    pointer += 12

    # read data section
    float_bytes = 8
    dtype_str = '>f' + str(float_bytes)
    float_buff = np.zeros([n_data_points])
    for i in range(n_data_points):
        byte_section = byte_array[pointer: pointer + float_bytes]
        float_buff[i] = np.frombuffer(byte_section, dtype=dtype_str)[0]
        pointer += float_bytes
    time_seq = 1 * float_buff  # seconds

    float_buff = np.zeros([n_data_points, 2])
    for i in range(n_data_points):
        byte_section = byte_array[pointer: pointer + float_bytes]
        float_buff[i, 0] = np.frombuffer(byte_section, dtype=dtype_str)[0]
        pointer += float_bytes
        byte_section = byte_array[pointer: pointer + float_bytes]
        float_buff[i, 1] = np.frombuffer(byte_section, dtype=dtype_str)[0]
        pointer += float_bytes
    float_buff = float_buff.T
    z_mag = 1 * float_buff[0]  # Ohm
    z_phs = 1 * float_buff[1]  # rad

    comments = byte_array[pointer:]
    comments_end = comments.find(b'\x00\x01')
    comments = comments[:comments_end]
    # unknown_data = byte_array[pointer + comments_end: pointer + comments_end + 3]
    pointer += comments_end + 3

    float_buff = np.zeros([n_data_points])
    for i in range(n_data_points):
        byte_section = byte_array[pointer: pointer + float_bytes]
        epoch_time = np.frombuffer(byte_section, dtype=dtype_str)[0]
        float_buff[i] = epoch_time
        pointer += float_bytes
    # Zahner epoch time is 10 years less than the standard epoch time starting 1970
    time_fmt = '%Y/%m/%d'
    time_difference = timegm(strptime('1980/01/01', time_fmt))
    time_stamp = float_buff + time_difference

    int_bytes = 2
    # significance of each frequency (data quality)
    for i in range(n_data_points):
        byte_section = byte_array[pointer: pointer + int_bytes]
        permille = int.from_bytes(byte_section, byteorder='big')
        float_buff[i] = permille / 1000.0
        pointer += int_bytes
    significance = 1 * float_buff

    # remaining_data = byte_array[pointer:]  # unidentified

    zx_data = np.vstack([time_seq, z_mag * np.cos(z_phs), z_mag * np.sin(z_phs), time_stamp, significance]).T

    if with_comments:
        comments = str(comments)[2:-1]  # str(bytes) -> b'xxxxxx'
        comments = comments.swapcase()
        comments = comments.replace('\\X00\\X03', '\nArea: ')
        comments = comments.replace('\\X00\\X06', 'Date(DDMMYY): ')
        comments = comments.replace('\\X00\\X04', '\n')
        comments = comments.replace('\\X00\\X05', '\n')
        comments = comments.replace('\\X00\\X07', '\n')
        comments = comments.replace('\\X00\\X08', '\n')
        comments = comments.replace('\\X00\\X09', '\n')
        comments = comments.replace('\\X00\\X0A', '\n')  # '\x0a' == '\n'
        comments = comments.replace('\\X00\\N', '\n')
        comments = comments.replace('\\X00\\X0B', '\n')
        comments = comments.replace('\\X00\\T', '\n')
        comments = comments.replace('\\X00\\X0C', '\n')
        comments = comments.replace('\\X00\\X0D', '\n')
        comments = comments.replace('\\X00\\X0E', '\n')
        comments = comments.replace('\\X00\\X0F', '\n')
        comments = comments.replace('\\X00\\X10', '\n')
        comments = comments.replace('\\X00\\X11', '\n')
        comments = comments.replace('\\X00\\X12', '\n')
        comments = comments.replace('\\X00\\X13', '\n')
        comments = comments.replace('\\X00\\X14', '\n')
        comments = comments.replace('\\X00\\X15', '\n')
        comments = comments.replace('\\X00\\X16', '\n')
        comments = comments.replace('\\X00\\X00', '')
        comments_end = comments.find('\\X00\\X01')
        if comments_end >= 0:
            comments = comments[comments.find('Date'): comments_end]  # discard unrecognized data
        else:
            comments = comments[comments.find('Date'):]  # discard unrecognized data
        comments = comments + '\n\n'
        titles = 'Variable' + title_delim + 'Z_real(Ohm)' + title_delim + 'Z_imag(Ohm)' + title_delim + \
                 'AbsTime(hhmmss.s*)' + title_delim + 'Significance'
        comments = comments + titles
        return zx_data, comments
    else:
        return zx_data


class ECM_ZView_IO:
    def __init__(self):
        # Conventions for ZView circuit model file *.mdl support
        # Do NOT modify unless ZView changes the conventions of its *.mdl model files
        mdl_header_dir = path.dirname(path.abspath(__file__))
        with open(path.join(mdl_header_dir, 'ZView_mdl_header.txt'), 'r') as header:
            self.zview_circuit_model_str = header.read()
        self.begin_cm_str = "Begin Circuit Model"
        self.end_cm_str = "End Circuit Model"
        self.begin_parallel_str = "Begin_Parallel"
        self.end_parallel_str = "End_Parallel"
        self.elem_str = "Element #"
        self.type_str = "Type"
        self.name_str = "Name"
        self.value_str = "Value"
        self.free_str = "Free"
        self.free_fixed = 0
        self.free_positive = 1
        self.free_arbitrary = 2
        self.type_begin_parallel = -1
        self.type_end_parallel = -2
        self.sub_column_pos = 2  # the sub column starts after 2 white spaces in a row (line)
        self.second_column_pos = 30  # the second column starts after 30 characters in a row (line)
        self.type_R = 1  # resistor
        self.type_C = 2  # capacitor
        self.type_L = 3  # inductor
        self.type_CPE = 11  # constant phase element
        self.type_GE = 12  # gerischer element
        self.type_QPE = 13

    def load_mdl_to_ecm_and_mecm(self, filename):
        # only effective if the mdl contains only Rs, L, and (Gerischers or RQs (cole element, CPE // R))
        from util_ecm import ECM, ManualECMPars
        with open(filename, 'r') as mdl_file:
            mdl_str_list = mdl_file.readlines()
        # bypass the zview circuit model section
        line_index = 0
        for i, line in enumerate(mdl_str_list):
            if self.begin_cm_str in line[:self.second_column_pos]:
                line_index = i
                break
        line_index += 1
        mdl_str_list = mdl_str_list[line_index:]  # leave only the circuit model section

        Rs = 0.0  # series resistor
        L = 0.0  # series inductor
        T = 0.0  # ZView model parameter
        P = 0.0  # ZView model parameter
        elem = {'R': 0.0, 'tau': 0.0, 'alpha': 0.0}
        elem_list = []
        isGerischer_list = []
        isFixed_list = []
        append_isFixed_list = False
        is_in_parallel = False
        type_flag = ''
        type_flag_lib = ('R', 'L', 'P', 'G', 'C', 'Q')
        # Resistor, Inductor, CPE, Gerischer, Capacitor, QPE, only effective in this method
        for line in mdl_str_list:
            # the lines are expected to be in the original order as exported by ZView
            col1 = line[:self.second_column_pos]
            if self.end_cm_str in col1:  # end of the circuit model section
                break
            if self.name_str in col1:
                continue  # skip name lines
            if self.elem_str in col1:  # an element
                # the order of lines for an element: type, (element)name, (parameter)name, value, free
                col2 = line[self.second_column_pos:]
                try:
                    if self.type_str in col1:  # element type info
                        if self.type_R == int(col2):  # resistor
                            type_flag = type_flag_lib[0]
                        elif self.type_L == int(col2):  # inductor
                            type_flag = type_flag_lib[1]
                        elif self.type_CPE == int(col2):  # constant phase element
                            type_flag = type_flag_lib[2]
                        elif self.type_GE == int(col2):  # gerischer
                            type_flag = type_flag_lib[3]
                        elif self.type_C == int(col2):  # capacitor
                            type_flag = type_flag_lib[4]
                        elif self.type_QPE == int(col2):  # Q constant phase element
                            type_flag = type_flag_lib[5]
                        elif self.type_begin_parallel == int(col2):
                            is_in_parallel = True
                        elif self.type_end_parallel == int(col2):
                            is_in_parallel = False
                        else:
                            raise ValueError("Unexpected element type")
                    elif self.value_str in col1:  # value info
                        if is_in_parallel:  # in parallel, is CPE or QPE (in current version)
                            # presumed read sequence: T, then P.
                            if type_flag == type_flag_lib[0]:  # resistor in series
                                elem['R'] = float(col2)
                            elif type_flag == type_flag_lib[2]:  # CPE
                                col1_split = col1.split(',')
                                if len(col1_split) > 1:
                                    par_id = int(col1_split[1].split(' ')[0])
                                    if par_id == 1:  # the 'T' parameter
                                        T = float(col2)
                                    elif par_id == 2:  # the 'P' parameter
                                        P = float(col2)
                                    else:
                                        raise ValueError("Unexpected par_id")
                            elif type_flag == type_flag_lib[4]:  # capacitor
                                T = float(col2)
                                P = 1.0
                            elif type_flag == type_flag_lib[5]:  # QPE
                                col1_split = col1.split(',')
                                if len(col1_split) > 1:
                                    par_id = int(col1_split[1].split(' ')[0])
                                    if par_id == 1:  # the 'T' parameter
                                        T = float(col2)
                                    elif par_id == 2:  # the 'P' parameter
                                        P = float(col2)
                                        T = np.power(T, P)
                                    else:
                                        raise ValueError("Unexpected par_id")
                            else:
                                raise ValueError("Unexpected type flag")
                            if elem['R'] > 0 and T > 0 and P > 0:
                                elem['tau'] = np.power(elem['R'] * T, 1. / P)
                                elem['alpha'] = P
                                elem_list.append(elem)
                                append_isFixed_list = True
                                elem = {'R': 0.0, 'tau': 0.0, 'alpha': 0.0}
                                isGerischer_list.append(False)
                                T = 0.0
                                P = 0.0
                        else:  # not in parallel
                            if type_flag == type_flag_lib[0]:  # resistor in series
                                Rs += float(col2)
                            elif type_flag == type_flag_lib[1]:  # inductor in series
                                L += float(col2)
                            elif type_flag == type_flag_lib[3]:  # Gerischer in series
                                col1_split = col1.split(',')
                                if len(col1_split) > 1:
                                    par_id = int(col1_split[1].split(' ')[0])
                                    if par_id == 1:  # the 'T' parameter
                                        T = float(col2)
                                    elif par_id == 2:  # the 'P' parameter
                                        P = float(col2)
                                    else:
                                        raise ValueError("Unexpected par_id")
                                    if T > 0 and P > 0:
                                        elem['R'] = 1.0 / (T * np.sqrt(P))
                                        elem['tau'] = 1.0 / P
                                        elem_list.append(elem)
                                        append_isFixed_list = True
                                        elem = {'R': 0.0, 'tau': 0.0, 'alpha': 0.0}
                                        isGerischer_list.append(True)
                                        T = 0.0
                                        P = 0.0
                            else:
                                raise ValueError("Unexpected type flag")
                    elif self.free_str in col1:
                        if append_isFixed_list:
                            append_isFixed_list = False
                            free_state = int(col2)
                            if free_state == self.free_fixed:
                                isFixed_list.append(True)
                            elif free_state == self.free_positive or free_state == self.free_arbitrary:
                                isFixed_list.append(False)
                            else:
                                raise ValueError("Unexpected free state")
                        else:
                            pass  # do nothing
                except ValueError:
                    raise ValueError("Failed reading ZView *.mdl file!")
            else:
                pass  # do nothing
        n_elem = len(elem_list)
        ecm = ECM(n_elem)
        mecm = ManualECMPars(n_elem)
        ecm.R_inf_ref = Rs
        ecm.L_ref = L
        for i, elem in enumerate(elem_list):
            ecm.rq_pars['R'][i] = elem['R']
            ecm.rq_pars['tau'][i] = elem['tau']
            ecm.rq_pars['alpha'][i] = elem['alpha']
            melem = mecm.get(i + 1)
            if isGerischer_list[i]:
                ecm.gerischer_ids.append(i)
                melem.isGerischer = True
            else:
                melem.isGerischer = False
            melem.fixed = isFixed_list[i]
            mecm.set(i + 1, melem)
        lntaus = np.log(ecm.rq_pars['tau'])
        lntaus_sorted = np.sort(lntaus)
        lntaus_sortarg = np.argsort(lntaus)
        if len(lntaus) > 1:
            lntau_distance = lntaus_sorted[1:] - lntaus_sorted[:-1]
        else:
            lntau_distance = []
        lntau_distance = np.insert(lntau_distance, 0, np.inf)
        lntau_distance = np.append(lntau_distance, np.inf)
        for i in range(n_elem):
            lntau_width = np.min([lntau_distance[i], lntau_distance[i+1]]) / 2.0
            mecm_id = lntaus_sortarg[i] + 1
            melem = mecm.get(mecm_id)
            melem.taumin = np.exp(lntaus_sorted[i] - lntau_width)
            melem.taumax = np.exp(lntaus_sorted[i] + lntau_width)
            mecm.set(mecm_id, melem)
        return ecm, mecm

    def save_ecm_as_mdl(self, filename, ecm, mecm=None):
        if filename is None:
            return
        Rs = ecm.R_inf_ref
        L = ecm.L_ref
        if hasattr(L, '__len__'):
            L = L[0]
        n_rq = ecm.n_rq
        isGerischer = [i in ecm.gerischer_ids for i in range(n_rq)]
        if mecm is None:
            isFixed = [True] * n_rq
        else:
            unfixed_indices = mecm.get_unfixed_indices()
            isFixed = [i not in unfixed_indices for i in range(n_rq)]

        # n_rq = parallel begins + R + Q + parallel ends, plus Rs and L
        n_G = np.sum(isGerischer)
        n_zview_elem = (n_rq - n_G) * 4 + n_G + 2

        zview_elem_id = [0]

        def add_R(text, R_val, elem_id, fixed=True):
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ' ' + self.type_str + ':').ljust(
                self.second_column_pos) \
                    + str(self.type_R) + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ' ' + self.name_str + ':').ljust(
                self.second_column_pos) \
                    + 'R' + str(elem_id) + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',0' + ' ' + self.name_str + ':').ljust(
                self.second_column_pos) \
                    + 'R' + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',0' + ' ' + self.value_str + ':').ljust(
                self.second_column_pos) \
                    + str(R_val) + '\n'
            if fixed:
                free_int = self.free_fixed
            else:
                free_int = self.free_positive
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',0' + ' ' + self.free_str + ':').ljust(
                self.second_column_pos) \
                    + str(free_int) + '\n'
            zview_elem_id[0] += 1
            return text

        def add_L(text, L_val, elem_id, fixed=True):
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ' ' + self.type_str + ':').ljust(
                self.second_column_pos) \
                    + str(self.type_L) + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ' ' + self.name_str + ':').ljust(
                self.second_column_pos) \
                    + 'L' + str(elem_id) + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',1' + ' ' + self.name_str + ':').ljust(
                self.second_column_pos) \
                    + 'T' + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',1' + ' ' + self.value_str + ':').ljust(
                self.second_column_pos) \
                    + str(L_val) + '\n'
            if fixed:
                free_int = self.free_fixed
            else:
                free_int = self.free_arbitrary
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',1' + ' ' + self.free_str + ':').ljust(
                self.second_column_pos) \
                    + str(free_int) + '\n'
            zview_elem_id[0] += 1
            return text

        def add_CPE(text, T_val, P_val, elem_id, fixed=True):
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ' ' + self.type_str + ':').ljust(
                self.second_column_pos) \
                    + str(self.type_CPE) + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ' ' + self.name_str + ':').ljust(
                self.second_column_pos) \
                    + 'CPE' + str(elem_id) + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',1' + ' ' + self.name_str + ':').ljust(
                self.second_column_pos) \
                    + 'T' + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',1' + ' ' + self.value_str + ':').ljust(
                self.second_column_pos) \
                    + str(T_val) + '\n'
            if fixed:
                free_int = self.free_fixed
            else:
                free_int = self.free_positive
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',1' + ' ' + self.free_str + ':').ljust(
                self.second_column_pos) \
                    + str(free_int) + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',2' + ' ' + self.name_str + ':').ljust(
                self.second_column_pos) \
                    + 'P' + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',2' + ' ' + self.value_str + ':').ljust(
                self.second_column_pos) \
                    + str(P_val) + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',2' + ' ' + self.free_str + ':').ljust(
                self.second_column_pos) \
                    + str(free_int) + '\n'
            zview_elem_id[0] += 1
            return text

        def add_GE(text, R0, tau0, elem_id, fixed=True):
            T = np.sqrt(tau0) / R0
            P = 1.0 / tau0
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ' ' + self.type_str + ':').ljust(
                self.second_column_pos) \
                    + str(self.type_GE) + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ' ' + self.name_str + ':').ljust(
                self.second_column_pos) \
                    + 'GE' + str(elem_id) + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',1' + ' ' + self.name_str + ':').ljust(
                self.second_column_pos) \
                    + 'T' + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',1' + ' ' + self.value_str + ':').ljust(
                self.second_column_pos) \
                    + str(T) + '\n'
            if fixed:
                free_int = self.free_fixed
            else:
                free_int = self.free_positive
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',1' + ' ' + self.free_str + ':').ljust(
                self.second_column_pos) \
                    + str(free_int) + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',2' + ' ' + self.name_str + ':').ljust(
                self.second_column_pos) \
                    + 'P' + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',2' + ' ' + self.value_str + ':').ljust(
                self.second_column_pos) \
                    + str(P) + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ',2' + ' ' + self.free_str + ':').ljust(
                self.second_column_pos) \
                    + str(free_int) + '\n'
            zview_elem_id[0] += 1
            return text

        def add_RQ(text, R0, tau0, alpha, elem_id, fixed=True):
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ' ' + self.type_str + ':').ljust(
                self.second_column_pos) \
                    + str(self.type_begin_parallel) + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ' ' + self.name_str + ':').ljust(
                self.second_column_pos) \
                    + self.begin_parallel_str + '\n'
            zview_elem_id[0] += 1
            text = add_R(text, R0, elem_id, fixed)
            T = np.power(tau0, alpha) / R0
            P = alpha
            text = add_CPE(text, T, P, elem_id, fixed)
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ' ' + self.type_str + ':').ljust(
                self.second_column_pos) \
                    + str(self.type_end_parallel) + '\n'
            text += ('  ' + self.elem_str + str(zview_elem_id[0]) + ' ' + self.name_str + ':').ljust(
                self.second_column_pos) \
                    + self.end_parallel_str + '\n'
            zview_elem_id[0] += 1
            return text

        text = ''
        text += self.zview_circuit_model_str + '\n'
        text += (self.begin_cm_str + ':').ljust(self.second_column_pos) + str(n_zview_elem) + '\n'
        text = add_R(text, Rs, 's', False)
        text = add_L(text, L, 1, False)
        for i in range(n_rq):
            if isGerischer[i]:
                text = add_GE(text, ecm.rq_pars['R'][i], ecm.rq_pars['tau'][i], i + 1, isFixed[i])
            else:  # is RQ
                text = add_RQ(text, ecm.rq_pars['R'][i], ecm.rq_pars['tau'][i], ecm.rq_pars['alpha'][i],
                              i + 1, isFixed[i])
        text += (self.end_cm_str + ':').ljust(self.second_column_pos) + str(n_zview_elem) + '\n'
        with open(filename, 'w+') as mdl_file:
            mdl_file.write(text)


def main():
    isw_filename = 'a.isw'
    isw_data = read_isw(isw_filename, with_comments=True)
    print(isw_data)


if __name__ == '__main__':
    main()
