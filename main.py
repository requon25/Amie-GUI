from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                             QHBoxLayout, QWidget, QPlainTextEdit, 
                             QPushButton, QLineEdit)

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QTextCursor

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

#Modelo
llm = ChatOllama(model="gemma3:4b")

MAX_TURNOS = 4


#Worker con STREAMING
class Worker(QThread):
    token = Signal(str)
    terminado = Signal()
    error = Signal(str)

    def __init__(self, llm, historial, pregunta):
        super().__init__()
        self.llm = llm
        self.historial = historial
        self.pregunta = pregunta

    def run(self):
        try:
            # agregar pregunta
            self.historial.append(HumanMessage(content=self.pregunta))

            #limitar historial
            if len(self.historial) > MAX_TURNOS * 2:
                self.historial[:] = self.historial[-MAX_TURNOS * 2:]

            respuesta_completa = ""

            #STREAMING
            for chunk in self.llm.stream(self.historial):
                if chunk.content:
                    self.token.emit(chunk.content)
                    respuesta_completa += chunk.content

            # guardar respuesta
            self.historial.append(AIMessage(content=respuesta_completa))

            self.terminado.emit()

        except Exception as e:
            self.error.emit(str(e))


#Ventana principal
class VentanaChatBot(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AMIE-ChatBot")
        self.resize(800, 500)

        self.historial = QPlainTextEdit()
        self.historial.setReadOnly(True) 
        
        self.entrada = QLineEdit() 
        self.entrada.setFixedHeight(40)
        self.entrada.setPlaceholderText("Escribe aqui...")

        self.boton_enviar = QPushButton("Enviar")
        self.boton_enviar.setFixedHeight(40)
        
        layout_entrada = QHBoxLayout()
        layout_entrada.addWidget(self.entrada)
        layout_entrada.addWidget(self.boton_enviar)

        layout_principal = QVBoxLayout()
        layout_principal.addWidget(self.historial)
        layout_principal.addLayout(layout_entrada)

        container = QWidget()
        container.setLayout(layout_principal)
        self.setCentralWidget(container)

        self.historial_chat = []
        self.worker = None

        self.boton_enviar.clicked.connect(self.enviar_mensaje)
        self.entrada.returnPressed.connect(self.enviar_mensaje)

    #enviar
    def enviar_mensaje(self):
        texto = self.entrada.text().strip()

        if not texto:
            return

        #evitar múltiples hilos
        if self.worker is not None and self.worker.isRunning():
            return

        self.historial.appendPlainText(f"USER: {texto}")
        self.entrada.clear()

        self.historial.appendPlainText("BOT: ")

        self.worker = Worker(llm, self.historial_chat, texto)

        self.worker.token.connect(self.agregar_token)
        self.worker.terminado.connect(self.fin_respuesta)
        self.worker.error.connect(self.mostrar_error)

        #IMPORTANTE: limpiar correctamente
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.limpiar_worker)

        if self.worker is not None:
            try:
                if self.worker.isRunning():
                    return
            except RuntimeError:
                self.worker = None

        self.worker.start()

    #agregar texto en tiempo real
    def agregar_token(self, token):
        self.historial.moveCursor(QTextCursor.End)
        self.historial.insertPlainText(token)

    #cuando termina
    def fin_respuesta(self):
        self.historial.appendPlainText("\n")

    #error
    def mostrar_error(self, error):
        self.historial.appendPlainText(f"\nERROR: {error}\n")
    
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
        event.accept()

    def limpiar_worker(self):
        self.worker = None

#run
if __name__ == "__main__":
    app = QApplication([])
    ventana = VentanaChatBot()
    ventana.show()
    app.exec()