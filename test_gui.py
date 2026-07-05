"""Quick test to verify PyQt5 can display a window."""
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt

app = QApplication(sys.argv)
w = QWidget()
w.setWindowTitle("AURA Test Window")
w.setGeometry(300, 200, 400, 200)
w.setStyleSheet("background-color: #0d1117;")

layout = QVBoxLayout(w)
label = QLabel("If you can see this, PyQt5 is working!")
label.setStyleSheet("color: #58a6ff; font-size: 20px; font-weight: bold;")
label.setAlignment(Qt.AlignCenter)
layout.addWidget(label)

w.show()
w.raise_()
w.activateWindow()
print("Window should be visible now. Close it to exit.")
sys.exit(app.exec_())
