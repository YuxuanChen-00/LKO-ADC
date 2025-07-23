import serial
import time

# 配置串口参数（根据实际串口号修改）
SERIAL_PORT = 'COM6'  # Windows示例：COM3；Linux示例：/dev/ttyACM0
BAUD_RATE = 115200  # 波特率需与Matlab一致

try:
    # 初始化串口
    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUD_RATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=2  # 读取超时时间（秒），可以适当延长以防响应慢
    )
    print(f"Python: 已连接串口 {SERIAL_PORT}，波特率 {BAUD_RATE}")

    # 持续发送数据并接收响应
    while True:
        # 准备发送的数据
        send_data = "Ping from Python " + time.strftime("%H:%M:%S")

        # 1. 记录发送前的时间戳
        start_time = time.time()

        # 发送数据到Matlab，并添加换行符
        ser.write(send_data.encode('utf-8') + b'\n')
        print(f"Python → Matlab: {send_data}")

        # 等待并读取Matlab响应
        received = ser.readline().decode('utf-8').strip()

        # 如果成功接收到响应
        if received:
            # 2. 记录接收到响应的时间戳
            end_time = time.time()
            # 3. 计算往返延迟（RTT），单位转换为毫秒
            latency_ms = (end_time - start_time) * 1000
            print(f"Matlab → Python: {received} | 往返延迟: {latency_ms:.2f} ms")
        else:
            # 如果在超时时间内没有收到数据
            print("Python: 未收到Matlab响应（超时）")

        time.sleep(1)  # 发送间隔

except Exception as e:
    print(f"错误: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("串口已关闭")
