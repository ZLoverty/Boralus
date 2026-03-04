import sys
import random
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QCheckBox, QGroupBox, QScrollArea)
from PySide6.QtCore import QTimer
import pyqtgraph as pg

class SixChannelDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("6通道工业传感器监控系统")
        self.resize(1200, 900)

        # 1. 扩展到 6 个变量的配置映射
        self.sensor_config = {
            'loadcell_01': {'label': '压力 1 (LC01)', 'color': '#FF5555', 'unit': 'kg'},
            'loadcell_02': {'label': '压力 2 (LC02)', 'color': '#FFAA00', 'unit': 'kg'},
            'rotary_01':   {'label': '转轴 1 (RE01)', 'color': '#55FF55', 'unit': 'deg'},
            'rotary_02':   {'label': '转轴 2 (RE02)', 'color': '#00AAFF', 'unit': 'deg'},
            'temp_01':     {'label': '环境温度 (T01)', 'color': '#FF55FF', 'unit': '°C'},
            'volt_01':     {'label': '系统电压 (V01)', 'color': '#FFFFFF', 'unit': 'V'},
        }

        self.max_points = 200
        self.data_store = {key: np.zeros(self.max_points) for key in self.sensor_config}
        self.ptr = 0 # 数据指针
        
        self.plot_items = {} # 存储 PlotItem (绘图区域)
        self.curves = {}     # 存储 PlotDataItem (线条)

        self.init_ui()

        # 2. 定时器更新
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_all_plots)
        self.timer.start(50) # 20Hz 更新频率

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- 左侧：控制开关 ---
        control_panel = QGroupBox("通道映射管理")
        control_layout = QVBoxLayout()
        for key, config in self.sensor_config.items():
            cb = QCheckBox(config['label'])
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {config['color']}; font-weight: bold;")
            cb.stateChanged.connect(lambda state, k=key: self.toggle_channel(k, state))
            control_layout.addWidget(cb)
        control_layout.addStretch()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(200)

        # --- 右侧：图形网格 (使用 GraphicsLayoutWidget 性能更好) ---
        self.win = pg.GraphicsLayoutWidget()
        
        # 自动创建 6 个绘图区，垂直排列
        last_plot = None
        for key, config in self.sensor_config.items():
            # 创建绘图项
            p = self.win.addPlot(title=config['label'])
            p.showGrid(x=True, y=True)
            p.setLabel('left', config['unit'])
            
            # 实现 X 轴联动：所有图表同步缩放
            if last_plot is not None:
                p.setXLink(last_plot)
            last_plot = p
            
            # 创建线条
            curve = p.plot(pen=pg.mkPen(config['color'], width=1.5))
            
            self.plot_items[key] = p
            self.curves[key] = curve
            
            # 换行：每个图表占一行
            self.win.nextRow()

        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.win)

    def toggle_channel(self, key, state):
        """动态显示/隐藏绘图区域"""
        plot_item = self.plot_items[key]
        if state == 2: # Checked
            plot_item.show()
            # 重新建立被破坏的联动关系（如果需要）
        else:
            plot_item.hide()

    def update_all_plots(self):
        """模拟 6 路数据同时涌入"""
        # 模拟后端收到的字典
        mock_packet = {
            'loadcell_01': 50 + random.uniform(-2, 2),
            'loadcell_02': 48 + random.uniform(-2, 2),
            'rotary_01':   (self.ptr * 5) % 360,
            'rotary_02':   (self.ptr * 3) % 360,
            'temp_01':     25 + np.sin(self.ptr / 10.0) * 2,
            'volt_01':     12 + random.uniform(-0.1, 0.1)
        }

        self.ptr += 1
        
        for key, value in mock_packet.items():
            if key in self.data_store:
                # 使用 numpy 滚动更新数据，效率极高
                self.data_store[key] = np.roll(self.data_store[key], -1)
                self.data_store[key][-1] = value
                
                # 只有在 PlotItem 可见时才更新渲染
                if self.plot_items[key].isVisible():
                    self.curves[key].setData(self.data_store[key])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SixChannelDashboard()
    window.show()
    sys.exit(app.exec())