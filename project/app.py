"""================================================================================

    app与web部分:网页与py文件交换数据

================================================================================"""


from flask import Flask, jsonify, render_template, request, Response
from flask_socketio import SocketIO, emit
import numpy as np
import random
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
from agent import llama_agent
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import socket
import threading
from langchain.tools import BaseTool
import time



IMAGE_FOLDER = "our_app/static/plot"

app = Flask(__name__)
socketio = SocketIO(app)

#一些全局变量
client_socket_global = None     #全局的socket
rfid_sig = 0.0
data_buffer = np.zeros([1, 10])
server_msg = "R"

#非常重要的一个函数！！！实施html的网页模板
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/data')
def get_data():
    temp_data = np.loadtxt('our_app/static/data.csv', delimiter=',', skiprows=2, encoding="utf-8", usecols=[6,7,8,9])
    global rfid_sig
    rfid_msg = "waiting"
    if rfid_sig == 1.0:
        rfid_msg = "检测到工作人员"
    elif rfid_sig == 2.0:
        rfid_msg = "检测到挖掘设备"
    elif rfid_sig == 3.0:
        rfid_msg = "检测到安全监测人员"
    elif rfid_sig == 4.0:
        rfid_msg = "检测到消防装置"
    elif rfid_sig == 5.0:
        rfid_msg = "检测到其他物品"
        

    data = {
        'MQ3': f"{temp_data[-1, 0]:.2f} ppm",
        'MQ135': f"{temp_data[-1, 1]:.2f} ppm",
        'MQ9': f"{temp_data[-1, 2]:.2f} ppm",
        'MQ2': f"{temp_data[-1, 3]:.2f} ppm",
        'RFID': rfid_msg,
        'mine_condition': '安全' if temp_data[-1, 0] < 800 and temp_data[-1, 1] < 800 and temp_data[-1, 2] < 800 and temp_data[-1, 3] < 800 else '危险'
    }
    return jsonify(data)


#plotly数据作图，实时上传到网页上
@app.route('/api/images')
def get_image():
    global data_buffer
    data_buffer = np.loadtxt('our_app/static/data.csv', delimiter=',', skiprows=2, encoding="utf-8", usecols=[0,1,2,3,6,7,8,9])
    data_buffer2 = np.loadtxt('our_app/static/data.csv', delimiter=',', skiprows=2, encoding="utf-8", usecols=[0,1,2,6,7,8,9])
    
    # 将数据转换为DataFrame
    df = pd.DataFrame(data_buffer, columns=['年', '月', '日', '时', 'MQ3', 'MQ135', 'MQ9', 'MQ2'])
    df2 = pd.DataFrame(data_buffer2, columns=['年', '月', '日', 'MQ3', 'MQ135', 'MQ9', 'MQ2'])
    
    # 对每个分组计算其他列的平均值
    hour_df = df.groupby(['年', '月', '日', '时']).mean().reset_index()
    day_df = df2.groupby(['年', '月', '日']).mean().reset_index()
    
    # 将结果转换回二维数组
    hour_data = hour_df.values
    day_data = day_df.values

    with open('our_app/static/day_data.txt', 'w') as file:
        # 遍历二维数组的每一行
        for row in day_data:
            # 将每一行的元素转换为字符串并用逗号分隔，然后写入文件并换行
            file.write(','.join(map(str, row)) + '\n')

    # 处理最近20小时的数据
    pro_time = [f"{int(row[1])}月{int(row[2])}日{int(row[3])}时" for row in hour_data]
    
    # 处理所有日期的数据
    pro_time2 = [f"{int(row[1])}月{int(row[2])}日" for row in day_data]
    
    # 创建图表
    fig = make_subplots(rows=1, cols=2, subplot_titles=("近小时数据监测结果", "近日数据监测结果"))
    
    # 图表1: 最近20小时的数据
    data1 = {
        'time': pro_time[-20:],
        'MQ3': hour_data[-20:, 4],
        'MQ135': hour_data[-20:, 5],
        'MQ9': hour_data[-20:, 6],
        'MQ2': hour_data[-20:, 7]
    }
    df1 = pd.DataFrame(data1)
    fig.add_trace(go.Scatter(x=df1['time'], y=df1['MQ3'], mode='lines', name='MQ3'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df1['time'], y=df1['MQ135'], mode='lines', name='MQ135'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df1['time'], y=df1['MQ9'], mode='lines', name='MQ9'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df1['time'], y=df1['MQ2'], mode='lines', name='MQ2'), row=1, col=1)
    
    # 图表2: 所有日期的数据
    if len(pro_time2)>20:
        data2 = {
            'time': pro_time2[-20:],
            'MQ3': day_data[-20:, 3],
            'MQ135': day_data[-20:, 4],
            'MQ9': day_data[-20:, 5],
            'MQ2': day_data[-20:, 6]
        }
    else:
        data2 = {
            'time': pro_time2[-len(pro_time2):],
            'MQ3': day_data[-len(pro_time2):, 3],
            'MQ135': day_data[-len(pro_time2):, 4],
            'MQ9': day_data[-len(pro_time2):, 5],
            'MQ2': day_data[-len(pro_time2):, 6]
        }
    df2 = pd.DataFrame(data2)
    fig.add_trace(go.Scatter(x=df2['time'], y=df2['MQ3'], mode='lines', name='MQ3'), row=1, col=2)
    fig.add_trace(go.Scatter(x=df2['time'], y=df2['MQ135'], mode='lines', name='MQ135'), row=1, col=2)
    fig.add_trace(go.Scatter(x=df2['time'], y=df2['MQ9'], mode='lines', name='MQ9'), row=1, col=2)
    fig.add_trace(go.Scatter(x=df2['time'], y=df2['MQ2'], mode='lines', name='MQ2'), row=1, col=2)
    
    # 设置图表的标题和轴标签
    fig.update_layout(
        title='各数据监测结果',
        xaxis_title='时间',
        yaxis_title='测量值/ppm'
    )
    
    graphJSON = pio.to_json(fig)
    return jsonify(graphJSON)



#上传agent的chat输出结果
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        return jsonify({'reply': '无效输入'}), 400
    bot_reply = test.chat(user_message)
    print(f"Model Reply: {bot_reply}")  # 添加日志输出
    return jsonify({'reply': bot_reply})


#上传保存过的agent的chat输出结果
@app.route('/api/get_file_content')
def init_summary():
    with open('our_app/static/bot_reply.txt', 'r') as file:
        file_content = file.read()  # 读取文件内容
    
    return jsonify({'message': file_content})



@app.route('/api/reload')
def summary():
    message = "Give me a comprehensive and huge analysis basing on these given data. Most importantly, The final answer must be at least 400 words long. The longer the final answer, the better!!!The longer the final answer, the better!!!The longer the final answer, the better!!! You mustn't use any tools."
    message = message + "The following is the average of the daily measurements of the mine, the first to last columns are 'year', 'month', 'day', 'alcohol concentration in the air', 'carbon dioxide, alcohol, benzene, nitrogen oxides, ammonia and other gases concentration', 'Carbon monoxide and other flammable gas concentration' and 'smoke density'."
    with open('our_app/static/day_data.txt', 'r') as file:
        file_content = file.read()  # 读取文件内容
    message = message + file_content
    
    bot_reply = test.chat(message)

    with open('our_app/static/bot_reply.txt', 'w') as output_file:
        output_file.write(bot_reply)

    return jsonify({'message': bot_reply})


#通过更改全局变量 server_msg ，将原本无效的socket_response回复覆盖为相应指令
@app.route('/control', methods=['POST'])
def control():
    global server_msg
    global client_socket_global
    data = request.json
    server_msg = data.get('key')
    # 在这里处理按键信号，例如控制你的设备
    print(f"Received key press: {server_msg}")
    
    # 向客户端发送数据
    if client_socket_global:
        try:
            client_socket_global.sendall(server_msg.encode('utf-8'))
        except Exception as e:
            print(f"Failed to send data to client: {e}")
    
    return jsonify(success=True)



@socketio.on('connect')
def handle_connect():
    print('Client connected')



@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')



@socketio.on('image')
def handle_image(data):
    # data 为 base64 编码的图像数据
    image_data = base64.b64decode(data)
    image = Image.open(BytesIO(image_data))
    image.save('received_image.png')  # 保存接收到的图像
    print('Image received and saved')


@socketio.on('data')
def handle_data(data):
    print('Data received and saved:',data)
    now = datetime.now()
    arr = np.array([now.year, now.month, now.day, now.hour, now.minute, now.second],dtype="float64")
    arr = arr.reshape((1, 6))
    
    global data_buffer
    data_buffer = np.loadtxt('our_app/static/data.csv', delimiter=',', skiprows=1, encoding="utf-8")
    if len(data_buffer.shape) == 1:
        data_buffer = data_buffer.reshape((1,10))
    
    data = np.fromstring(data, dtype=float, sep=",")
    data = data.reshape((1,data.shape[0]))
    
    processed_data = np.concatenate([arr,data],axis=1)
    data_buffer = np.concatenate([data_buffer,processed_data],axis=0)
    
    socketio.emit('response', {'status': 'success'})  # 发送回执给客户端
 
    with open("our_app/static/data.csv", 'r', encoding="utf-8") as f:
        header = f.readline().strip()
    with open("our_app/static/data.csv", 'w', encoding="utf-8") as f:
        np.savetxt(f, data_buffer, delimiter=',', fmt='%f', header=header, comments='')
    

def translate(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_text


def start_socketio():
    socketio.run(app, host="0.0.0.0",port=3000, debug=False)


def handle_client(client_socket):
    
    with client_socket:
        while True:
            time1 = time.time()
            while True:
                time2 = time.time()
                global server_msg
                client_msg = client_socket.recv(1024)
                if not client_msg:
                    break

                client_msg = client_msg.decode('utf-8', errors='ignore')
                client_socket.sendall(server_msg.encode('utf-8'))
                server_msg = "R"
                # print(f"Received: {client_msg}")

                if len(client_msg) > 8 and time2-time1 > 5:
                    print(f"Received: {client_msg}")
                    now = datetime.now()
                    arr = np.array([now.year, now.month, now.day, now.hour, now.minute, now.second], dtype="float64")
                    arr = arr.reshape((1, 6))
                    
                    #with data_lock:
                    data_buffer = np.loadtxt('our_app/static/data.csv', delimiter=',', skiprows=1, encoding="utf-8")
                    if len(data_buffer.shape) == 1:
                        data_buffer = data_buffer.reshape((1, 10))
                    
                    client_msg = np.fromstring(client_msg, dtype=float, sep=",")
                    if client_msg.shape[0] > 5:
                        continue

                    global rfid_sig
    
                    rfid_sig = client_msg[-1]
                    client_msg = client_msg[:-1]
                    client_msg = client_msg.reshape((1, client_msg.shape[0]))
                    
                    print(rfid_sig)
                    print(client_msg)
                    
                    processed_data = np.concatenate([arr, client_msg], axis=1)
                    data_buffer = np.concatenate([data_buffer, processed_data], axis=0)
                    
                    with open("our_app/static/data.csv", 'r', encoding="utf-8") as f:
                        header = f.readline().strip()
                    with open("our_app/static/data.csv", 'w', encoding="utf-8") as f:
                        np.savetxt(f, data_buffer, delimiter=',', fmt='%f', header=header, comments='')
                        print("Writing Done")
                    break


def start_socket():
    host = '0.0.0.0'
    port = 9999
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        client_handler = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler.start()



"""================================================================================

    Agent部分:自定义tools

================================================================================"""



class Warning(BaseTool):
    name = "Warn Everyone To Leave"
    description = "When the 'mine condition' is examined to be 'dangerous'. Use this tools to warn everyone to leave out the mine."
 
    def _run(self, input: str) -> str:
        global server_msg
        server_msg="Z"
        # 向客户端发送数据
        if client_socket_global:
            try:
                client_socket_global.sendall(server_msg.encode('utf-8'))
            except Exception as e:
                print(f"Failed to send data to client: {e}")
        print("Tools 1 works well and free!")
        # Your logic here
        return "Warning minesmen to leave "
 
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    


class Moving_forward(BaseTool):
    name = "Robot move forward"
    description = "This is a custom tool for you to make the robot keep moving forward but it will not stop. Don't use it unless it is needed"
 
    def _run(self, input: str) -> str:
        global server_msg
        server_msg = "C"
        # # 向客户端发送数据
        if client_socket_global:
            try:
                client_socket_global.sendall(server_msg.encode('utf-8'))
            except Exception as e:
                print(f"Failed to send data to client: {e}")
        print("Tools 2 works well and free!")
        # # Your logic here
        return "Robot is moving forward successfully. You need another tools to stop it."
 
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    

    
class Rotate(BaseTool):
    name = "Robot rotate"
    description = "This is a custom tool for you to set the robot's forward speed to 10cm per second. Don't use it unless it is needed"
 
    def _run(self, input: str) -> str:
        global server_msg
        server_msg = "V"
        # 向客户端发送数据
        if client_socket_global:
            try:
                client_socket_global.sendall(server_msg.encode('utf-8'))
            except Exception as e:
                print(f"Failed to send data to client: {e}")
        print("Tools 3 works well and free!")
        time.sleep(3)
        # Your logic here
        return "Robot is rotating around successfully"
 
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

class Stop_Robot(BaseTool):
    name = "stop the robot"
    description = "This is a custom tool for you to make the robot stop. Use it if you need the robot stop."
 
    def _run(self, input: str) -> str:
        global server_msg
        server_msg = "Q"
        # 向客户端发送数据
        if client_socket_global:
            try:
                client_socket_global.sendall(server_msg.encode('utf-8'))
            except Exception as e:
                print(f"Failed to send data to client: {e}")
        print("Tools 3 works well and free!")
        # Your logic here
        return "Robot has been stopped successfully"
 
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")   
    

class check_mine(BaseTool):
    name = "check mine condition"
    description = "This is a custom tool for you to check the mine's condition. You can not use this tool to get related information unless user ask for the condition of the mine."
 
    def _run(self, input: str) -> str:
        message = "The following is the average of the daily measurements of the mine, from the first to last columns are 'year', 'month', 'day', 'alcohol concentration in the air', 'carbon dioxide, alcohol, benzene, nitrogen oxides, ammonia and other gases concentration', 'Carbon monoxide and other flammable gas concentration' and 'smoke density'."
        with open('our_app/static/day_data.txt', 'r') as file:
            file_content = file.read()  # 读取文件内容
        message = message + file_content

        
        print("Tools 4 works well and free!")
        # Your logic here
        return str(message)
 
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async") 
    

if __name__ == '__main__':

    test=llama_agent()
    test.instructions = """Remeber these words. You're a professional mine data analyst.
        Your responsibility is to analyze the mine situation for the user based on the existing data, 
        including but not limited to various data trends, safety risks, construction recommendations, etc.
        Please consider whether you should use the tools to answer the users' question or requiremnet!
        By the way, you can still chat with users as a friend if it is needed. Give me anwser as long as you can."""
    test.set_tools([Moving_forward(),Rotate(),Stop_Robot(),check_mine()])
    test.create_agent()
    
    socketio_server = threading.Thread(target=start_socketio)
    socket_server = threading.Thread(target=start_socket)

    socketio_server.start()
    socket_server.start()

    socketio_server.join()
    socket_server.join()
    

