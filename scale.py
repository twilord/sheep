import serial
import re
import rembg


class Scale():
    def __init__(self, com):
        self.com = serial.Serial(
            com,
            baudrate=9600,
            timeout=1,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )

    def read(self):
        if self.com.isOpen():
            self.com.reset_input_buffer()
            data = self.com.readline()
            data.decode('ascii', 'ignore')
            data = str(data, 'utf-8')
            data = data.replace('\n', '').replace('\r', '')
            number = re.sub(r'[A-Za-z]', "", data)
            if len(number) > 1:
                return float(number)
            else:
                return 0.0
