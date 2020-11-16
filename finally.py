import socket
import struct
import numpy as np
import tensorflow as tf

# 2.连接服务器
dest_ip = "192.168.1.110"
dest_port =5000
dest_addr = (dest_ip, dest_port)

model=tf.keras.models.load_model("./model.h5")
gettus=(
    "1 --> 右手 弯曲",
    "2 --> 左手 弯曲",
    "3 --> 右手 翻转",
    "4 --> 左手 翻转",
    "5 --> 右手 前伸",
    "6 --> 左手 前伸",
)
tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_socket.connect(dest_addr)
try:
    while 1:
        recv_data = tcp_socket.recv(3094)

        tcp_socket.send("OK".encode('utf-8'))
        #print(recv_data) 
        rx_buffer=recv_data
        count=0
        tensor=[]
        for i in range(119):
            if rx_buffer[0+i*26:2+i*26]==b'\xaa\xbb':
                tensor += list(struct.unpack("6f", rx_buffer[2+i*26:26+i*26]))
                count=count+1
        if count ==119:
           # print("recv OK")
            tensor=((np.array(tensor)+2000)/4000)
            inputs = np.array(tensor).reshape(1,714)
            predictions= model.predict(inputs)
            a=np.round(predictions, decimals=3)
            print(a)
            print(gettus[np.argmax(a)])
         
except KeyboardInterrupt:
    #4. 关闭套接字socket
    tcp_socket.close()  
    print("exit")
    pass

