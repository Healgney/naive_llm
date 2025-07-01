import socket
import threading

def check_ssh(ip):
    try:
        sock = socket.create_connection((ip, 22), timeout=0.5)
        print(f"SSH open: {ip}")
        sock.close()
    except:
        pass

base_ip = "192.168.1."  # 替换为你自己的网段
threads = []

for i in range(1, 255):
    ip = base_ip + str(i)
    t = threading.Thread(target=check_ssh, args=(ip,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
