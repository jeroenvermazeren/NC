"""Build the tkinter gui root"""
import math
from PyQt5.QtWidgets import *#(QWidget, QToolTip, QDesktopWidget, QPushButton, QApplication)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal, pyqtSlot 
from PyQt5.QtGui import QIntValidator, QDoubleValidator
import sys
from PSO_system.Counting.plot import PlotCanvas
from PSO_system.Counting.run import CarRunning
from PSO_system.Counting.test_result import TestRunning

THREADS = []

class GuiRoot(QWidget):
    """Root of gui."""
    def __init__(self, dataset, training_data):
        """Create GUI root with datasets dict"""
        super().__init__()
        self.threadpool = QThreadPool()
        self.setFixedSize(800, 800)
        self.center()
        self.setWindowTitle('PSO')      
        self.show()
        #read the map and training data
        self.map_datalist = dataset.keys()
        self.map_data = dataset
        self.training_datalist = training_data.keys()
        self.training_data = training_data

        #creat file choosing area
        self.file_run_creation(self.map_datalist, self.training_datalist)
        
        self.operation_parameter_creation()
        self.ouput_text_creation()
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        vbox.addWidget(self.file_run)
        vbox.addWidget(self.operation_type)
        vbox.addWidget(self.text_group_box)
        hbox.addLayout(vbox)
        self.m = PlotCanvas(self.map_data)
        hbox.addWidget(self.m)
        self.setLayout(hbox)
    def file_run_creation(self, datalist, training_data):
        self.file_run = QGroupBox("File choose")
        layout = QGridLayout()
        layout.setSpacing(10)
        map_file_label = QLabel("Map file: ")

        self.map_file_choose = QComboBox()
        for i in datalist:
            self.map_file_choose.addItem("{}".format(i))
        self.map_file_choose.currentTextChanged.connect(self.file_changed)

        training_file_label = QLabel("Training file: ")
        self.training_file_choose = QComboBox()
        for i in training_data:
            self.training_file_choose.addItem("{}".format(i))
        self.run_btn = QPushButton("Start", self)
        self.run_btn.clicked.connect(self.run)

        self.test_btn = QPushButton("Test", self)
        self.test_btn.clicked.connect(self.test_rbfn)
        layout.addWidget(map_file_label, 1, 0, 1, 1)
        layout.addWidget(self.map_file_choose, 1, 1, 1, 3)
        layout.addWidget(training_file_label, 2, 0, 1, 1)
        layout.addWidget(self.training_file_choose, 2, 1, 1, 3)
        layout.addWidget(self.run_btn, 3, 0, 1, 4)
        layout.addWidget(self.test_btn, 4, 0, 1, 4)
        layout.setVerticalSpacing(0)
        layout.setHorizontalSpacing(0)
        self.file_run.setLayout(layout)

        self.test_parameter = None
    def operation_parameter_creation(self):
        """Operation parameter field"""
        self.operation_type = QGroupBox("Operation parameter setting")
        vbox = QVBoxLayout()
        
        #Set and operation paremeter region, including iteration times, population number, 
        #mutation probability, crossover probability, network j value

        textfield_layout = QHBoxLayout()
        textfield_setting = QLabel("File Name:")
        self.textfield_line = QLineEdit()
        # self.textfield_line.setValue('Hello')
        self.textfield_line.setPlaceholderText("results")
        self.textfield_line.setMaximumWidth(150)
        textfield_layout.addWidget(textfield_setting)
        textfield_layout.addWidget(self.textfield_line)
        textfield_layout.insertSpacing(-1,100)


        iteration_layout = QHBoxLayout()
        iteration_setting = QLabel("Maximum number of iterations :")
        self.iteration_line = QSpinBox()
        self.iteration_line.setRange(1, 10000)
        self.iteration_line.setValue(40)
        self.iteration_line.setMaximumWidth(150)
        iteration_layout.addWidget(iteration_setting)
        iteration_layout.addWidget(self.iteration_line)
        iteration_layout.insertSpacing(-1,100)

        ant_colony_size_layout = QHBoxLayout()
        ant_colony_size_setting = QLabel("Ant colony size:")
        self.ant_colony_size_line = QSpinBox()
        self.ant_colony_size_line.setRange(2, 10000)
        self.ant_colony_size_line.setValue(2)
        self.ant_colony_size_line.setMaximumWidth(150)
        ant_colony_size_layout.addWidget(ant_colony_size_setting)
        ant_colony_size_layout.addWidget(self.ant_colony_size_line)
        ant_colony_size_layout.insertSpacing(-1,100)

        archive_size_layout = QHBoxLayout()
        archive_size = QLabel("Archive size: ")
        self.archive_size_line = QSpinBox()
        self.archive_size_line.setRange(2, 100)
        # self.archive_size_line.setDecimals(2)
        self.archive_size_line.setValue(50)
        self.archive_size_line.setMaximumWidth(150)
        archive_size_layout.addWidget(archive_size)
        archive_size_layout.addWidget(self.archive_size_line)
        archive_size_layout.insertSpacing(-1,100)

        # in PSO, φ1 means the parameter multiplied with (pi(t)-x(t))
        locality_layout = QHBoxLayout()
        locality_setting = QLabel("Locality of search: ")
        self.locality_line = QDoubleSpinBox()
        self.locality_line.setValue(0.85)
        self.locality_line.setRange(0, 1)
        self.locality_line.setDecimals(2)
        self.locality_line.setMaximumWidth(150)
        locality_layout.addWidget(locality_setting)
        locality_layout.addWidget(self.locality_line)
        locality_layout.insertSpacing(-1,100)

        convergence_speed_layout = QHBoxLayout()
        convergence_speed_setting = QLabel("Speed of convergence: ")
        self.convergence_speed_line = QDoubleSpinBox()
        self.convergence_speed_line.setRange(0, 1)
        self.convergence_speed_line.setDecimals(4)
        self.convergence_speed_line.setValue(0.001)
        self.convergence_speed_line.setMaximumWidth(150)
        convergence_speed_layout.addWidget(convergence_speed_setting)
        convergence_speed_layout.addWidget(self.convergence_speed_line)
        convergence_speed_layout.insertSpacing(-1,100)

        net_j_layout = QHBoxLayout()
        net_j_setting = QLabel("Network neurl number j: ")
        self.net_j_line = QSpinBox()
        self.net_j_line.setRange(1,20)
        self.net_j_line.setValue(6)
        self.net_j_line.setMaximumWidth(150)
        net_j_layout.addWidget(net_j_setting)
        net_j_layout.addWidget(self.net_j_line)
        net_j_layout.insertSpacing(-1,100)

        sd_layout = QHBoxLayout()
        sd_setting = QLabel("Maximum SD: ")
        self.sd_line = QSpinBox()
        self.sd_line.setRange(1,100)
        self.sd_line.setValue(10)
        self.sd_line.setMaximumWidth(150)
        sd_layout.addWidget(sd_setting)
        sd_layout.addWidget(self.sd_line)
        sd_layout.insertSpacing(-1,100)

        v_max_layout = QHBoxLayout()
        v_max_setting = QLabel("Maximum Velocity: ")
        self.v_max_line = QDoubleSpinBox()
        self.v_max_line.setRange(0, 10)
        self.v_max_line.setDecimals(2)
        self.v_max_line.setValue(4)
        self.v_max_line.setMaximumWidth(150)
        v_max_layout.addWidget(v_max_setting)
        v_max_layout.addWidget(self.v_max_line)
        v_max_layout.insertSpacing(-1,100)

        vbox.addLayout(textfield_layout)
        vbox.addLayout(iteration_layout)
        vbox.addLayout(ant_colony_size_layout)
        vbox.addLayout(archive_size_layout)
        vbox.addLayout(locality_layout)
        vbox.addLayout(convergence_speed_layout)
        vbox.addLayout(net_j_layout)
        # vbox.addLayout(v_max_layout)
        vbox.addLayout(sd_layout)
        self.operation_type.setLayout(vbox)
        
    def ouput_text_creation(self):
        self.text_group_box = QGroupBox("Execution log")
        layout = QVBoxLayout()
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        layout.addWidget(self.console)
        self.text_group_box.setLayout(layout)
    def file_changed(self):
        """print map"""
        self.m.plot_map(self.map_file_choose.currentText())
        self.console.append('Map changed')
    def run(self):
        self.test_parameter = None
        l = []
        l.append(self.iteration_line.value())
        l.append(self.ant_colony_size_line.value())
        l.append(self.archive_size_line.value())
        l.append(self.locality_line.value())
        l.append(self.convergence_speed_line.value())
        l.append(self.net_j_line.value())
        l.append(0)
        l.append(self.sd_line.value())
        l.append(self.textfield_line.text())

        # disable avoid to touch
        self.disable('yes')

        # transfer for counting
        for i in range(1):
            print("\n\nITERATION = ", i)
            self.console.append('Start training RBFN with ACO')
            car = CarRunning(self.map_data, self.map_file_choose.currentText(), self.training_data, self.training_file_choose.currentText(), l,i)
            car.signals.iteration.connect(self.console_output)
            car.signals.result.connect(self.dir_test_rbfn)
            self.threadpool.start(car)
            # self.threadpool.waitForDone()
            # self.threadpool.start(self.test_rbfn())
            # self.threadpool.waitForDone()

    def dir_test_rbfn(self, parameters):
        # disable avoid to touch
        self.disable('yes')
        # transfer for counting
        self.test_parameter = parameters
        self.console.append('Start testing result on current map.')
        self.console.append("------------------------------------------------------")
        test_thread = TestRunning(self.map_data, self.map_file_choose.currentText(), parameters, None)
        test_thread.signals.plot.connect(self.plot_output)
        self.threadpool.start(test_thread)
    def test_rbfn(self):
        if self.test_parameter == None:
            self.console.append('No RBFN model, please push [Start] button first.')
        else:
            # disable avoid to touch
            self.disable('yes')
            # transfer for counting
            self.console.append('Start testing result on current map.')
            self.console.append("------------------------------------------------------")
            test_thread = TestRunning(self.map_data, self.map_file_choose.currentText(), None, self.test_parameter)
            test_thread.signals.plot.connect(self.plot_output)
            self.threadpool.start(test_thread)
    def console_output(self, s):
        self.console.append(str(s))
    def plot_output(self, s):
        self.m.plot_car(s)
        self.disable('no')
        self.console.append('Test is complete, and showing on right area')
        self.console.append("------------------------------------------------------")
    def center(self):
        """Place window in the center"""
        qr = self.frameGeometry()
        central_p = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(central_p)
        self.move(qr.topLeft())
    def disable(self, yes_or_no):
        if yes_or_no == 'yes':
            self.iteration_line.setDisabled(True)
            self.ant_colony_size_line.setDisabled(True)
            self.archive_size_line.setDisabled(True)
            self.convergence_speed_line.setDisabled(True)
            self.locality_line.setDisabled(True)
            self.net_j_line.setDisabled(True)
            self.map_file_choose.setDisabled(True)
            self.training_file_choose.setDisabled(True)
            self.run_btn.setDisabled(True)
            self.test_btn.setDisabled(True)
            self.v_max_line.setDisabled(True)
            self.sd_line.setDisabled(True)
        else:
            self.iteration_line.setDisabled(False)
            self.ant_colony_size_line.setDisabled(False)
            self.archive_size_line.setDisabled(False)
            self.convergence_speed_line.setDisabled(False)
            self.locality_line.setDisabled(False)
            self.net_j_line.setDisabled(False)
            self.map_file_choose.setDisabled(False)
            self.training_file_choose.setDisabled(False)
            self.run_btn.setDisabled(False)
            self.test_btn.setDisabled(False)
            self.v_max_line.setDisabled(False)
            self.sd_line.setDisabled(False)
if __name__ == '__main__':
    print("Error: This file can only be imported. Execute 'main.py'")
