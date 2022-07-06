import serial
import time
import sys
import binascii


class DataPack:
    def __init__(self):
        self.head = 'A0'

    @staticmethod
    def _get_check(_body: str) -> str:
        _body_index = [i for i in range(0, len(_body), 2)]
        _body_byte = [str(_body[i] + _body[i+1]) for i in _body_index]
        _check = 0
        for _index, _byte in enumerate(_body_byte):
            _check += int(_byte, base=16)
        _check = 256 - (_check % 256)
        _check_str = hex(_check).replace('0x', '').upper()
        return _check_str

    @staticmethod
    def _get_len(str_to_get_len: str):
        _len = hex(int(len(str_to_get_len) / 2)).replace('0x', '').upper()
        if len(_len) == 1:
            return '0' + _len
        else:
            return _len

    # 发送选择命令的标签的epc
    def pack_set_access_epc_match(self, epc: str) -> str:
        address = '01'
        cmd = '85'
        mode = '00'
        _body = address + cmd + mode + DataPack._get_len(epc) + epc
        _body = _body + DataPack._get_check(_body)
        _body = self.head + DataPack._get_len(_body) + _body
        return _body


def data_unpack(data: str) -> dict:
    # data_index = [i for i in range(0, len(data), 2)]
    # _str = [str(data[i]+data[i+1]) for i in data_index]
    # _str_int = []
    # for _index, _hex in enumerate(_str):
    #     _str_int.append(int(_hex, base=16))
    # print(_str_int)
    _data_dict = {
        'rssi': data[12:14],
        'epc': data[20:44],
        'tid': data[44:68]
    }
    print(_data_dict)
    return _data_dict


class UhfRfid:
    def __init__(self, com: str):
        self.com = serial.Serial(
            com,
            baudrate=115200,
            timeout=1,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )
        self.read = bytes.fromhex('A006 018B 0000 01CD')
        self.read_8c = bytes.fromhex('A00A FF8C 0000 0100 0600 00C4')
        self.set_work_antenna = {
            '1': bytes.fromhex('A004 0174 00E7'),
            '2': bytes.fromhex('A004 0174 01E6'),
            '3': bytes.fromhex('A004 0174 02E5'),
            '4': bytes.fromhex('A004 0174 03E4')}
        if self.com.isOpen():
            print('Print Message: open success', 'File: "' + __file__ + '", Line ' + str(sys._getframe().f_lineno))

    def is_open(self) -> bool:
        if self.com.isOpen():
            return True
        else:
            return False

    def com_set_work_antenna(self, number: str) -> (bool, str):
        if not self.com.isOpen():
            return False, ''
        if number in ['1', '2', '3', '4']:
            self.com.write(self.set_work_antenna[number])
            _read_back = self.com.read(100)
            return True, _read_back.hex()
        return False, ''

    def com_customized_session_target_inventory(self) -> (bool, str):
        if not self.com.isOpen():
            return False, ''
        self.com.write(self.read)
        _read_back = self.com.read(100)
        return True, _read_back.hex()

    def com_set_access_epc_match(self, epc: str) -> (bool, str):
        if not self.com.isOpen():
            return False, ''
        _str_head = 'A011 0185'
        _str_body = (_str_head + epc).replace(' ', '')
        _check = DataPack._get_check(_str_body)
        _str_body = _str_body + _check
        return True, _str_body

    def com_inventory_epc_tid_user(self) -> (bool, str):
        if not self.com.isOpen():
            return False, ''
        self.com.write(self.read_8c)
        _read_back = self.com.read(100)
        return True, _read_back


if __name__ == '__main__':
    uhf_rfid = UhfRfid('COM5')
    ret, read_back = uhf_rfid.com_set_work_antenna('4')
    # print(ret, read_back)
    ret, read_back = uhf_rfid.com_inventory_epc_tid_user()
    # print(ret, read_back)
    # print(read_back.hex().upper())
    data_unpack(read_back.hex())

