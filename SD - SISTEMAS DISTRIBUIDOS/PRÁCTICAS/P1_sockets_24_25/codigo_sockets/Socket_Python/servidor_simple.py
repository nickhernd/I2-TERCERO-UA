import socket

HOST = 'localhost'
PORT = 8010

def invertir_cadena(cadena):
    return cadena[::-1]

# Creación del socket del servidor
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Servidor escuchando en {HOST}:{PORT}")
    
    while True:
        conn, addr = s.accept()
        with conn:
            print(f"Conexión establecida desde {addr}")
            data = conn.recv(1024).decode('utf-8')
            if not data:
                break
            print(f"Recibido: {data}")
            
            # Invertir la cadena recibida
            respuesta = invertir_cadena(data)
            
            print(f"Enviando: {respuesta}")
            conn.sendall(respuesta.encode('utf-8'))

    print("Servidor cerrado")