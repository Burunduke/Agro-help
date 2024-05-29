import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
import configparser
import pathlib
from osgeo import gdal, gdal_array
from matplotlib.widgets import LassoSelector
from matplotlib import path
from torchvision import transforms
from dataloaders import custom_transforms as tr
from dataloaders.helpers import *
from networks.mainnetwork import *
import segmentation_refinement as refine
import cv2
import matplotlib.pyplot as plt
import os
from math import sqrt


def chose():
    plt.rcParams['toolbar'] = 'None'
    data = np.load('temp/rgb_array.npy')
    fig = plt.figure('Выберите интересующий регион', frameon=False)
    this_manager = plt.get_current_fig_manager()
    this_manager.window.setWindowIcon(QtGui.QIcon((os.path.join('ims', 'toolbox.ico'))))
    plt.axis('off')

    ax1 = fig.add_subplot(121)
    ax1.imshow(data)
    ax1.axis('off')

    ax2 = fig.add_subplot(122)
    msk = ax2.imshow(np.zeros(data.shape), vmax=1, interpolation='nearest')
    ax2.axis('off')
    plt.subplots_adjust()

    # Pixel coordinates
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    pix = np.vstack((x.flatten(), y.flatten())).T

    def on_select(verts):
        p = path.Path(verts)
        ind = p.contains_points(pix, radius=1)
        ind = ind.reshape((data.shape[0], data.shape[1])) # (w, h)

        mask = np.where(ind, 1, 0) # inside = 1, outside = 0
        np.save('temp/mask_roi.npy', mask) # save

        ind = ind[:, :, None] # (w, h, )
        selected = np.broadcast_to(ind, data.shape) #(w, h, 3)
        selected = np.where(selected, data, 1)
        msk.set_data(selected)
        np.save('temp/rgb_roi.npy', selected)
        fig.canvas.draw_idle()

    lasso = LassoSelector(ax1, on_select)

    plt.tight_layout()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()


class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config_sensors = configparser.ConfigParser()
        self.config_sensors.read("Sensors.ini")
        self.sensors = list(dict(self.config_sensors.items('Sensors')).keys())  # List of sensors

        self.config_formulas = configparser.ConfigParser()
        self.config_formulas.read("Formulas.ini")
        self.complex = list(dict(self.config_formulas.items('Complex indices')).keys())  # List of sensors
        self.vegetation = list(dict(self.config_formulas.items('Vegetation indices')).keys())  # List of sensors
        self.water = list(dict(self.config_formulas.items('Water indices')).keys())  # List of sensors
        self.geo = list(dict(self.config_formulas.items('Geo indices')).keys())  # List of sensors
        self.burn = list(dict(self.config_formulas.items('Burn indices')).keys())  # List of sensors

        self.initUI()

        self.setWindowTitle("Помощь агроному")
        self.setWindowIcon(QtGui.QIcon((os.path.join('ims', 'toolbox.ico'))))

        # QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.showMaximized()
        qApp.processEvents()
        if state == 1:
            self.sensors_again()

    def initUI(self):
        main_layout = QGridLayout(self)
        self.setLayout(main_layout)

        self.setWindowTitle("VI calculator")  # Задаем главное окно

        self.tabWidget = QTabWidget()  # Задаем виджет для вкладок

        self.tab_1 = QWidget(self)  # Создаем 1 вкладку

        self.tab_1_horiz = QHBoxLayout()

        self.tab_1_left_vert = QVBoxLayout()

        # file selector
        self.file_layout = QHBoxLayout()
        self.file_browse = QPushButton('Открыть')
        self.file_browse.clicked.connect(self.open_file_dialog)
        self.filename_edit = QLineEdit()

        self.file_layout.addWidget(QLabel('Файл:'))
        self.file_layout.addWidget(self.filename_edit)
        self.file_layout.addWidget(self.file_browse)

        self.tab_1_left_vert.addLayout(self.file_layout)

        self.chose_resize_rate = QLabel()
        self.chose_resize_rate.setText(
            "Выберите степень сжатия (чем выше, тем меньшее количество ОЗУ и видеопамяти требуется.\n"
            "Также ускоряет дальнейшую обработку, но снижает точность.\n"
            "При уточнении границ и вычислении индексов изображение будет возвращено к изначальному размеру")
        self.tab_1_left_vert.addWidget(self.chose_resize_rate)

        self.selector_resize = QComboBox()
        self.selector_resize.addItems(['1', '2', '4', '8'])
        if state == 0:
            self.selector_resize.setEnabled(False)
        self.tab_1_left_vert.addWidget(self.selector_resize)

        self.button_resize = QPushButton()
        self.button_resize.setText("Сжать изображение")
        if state == 0:
            self.button_resize.setEnabled(False)
        self.button_resize.clicked.connect(self.button_resize_image)
        self.scale_const = 1
        self.tab_1_left_vert.addWidget(self.button_resize)

        self.progress = QProgressBar()
        self.tab_1_left_vert.addWidget(self.progress)

        self.chose_sensor = QLabel()
        self.chose_sensor.setText("Выберите сенсор, на который была произведена съемка")
        self.tab_1_left_vert.addWidget(self.chose_sensor, alignment=QtCore.Qt.AlignCenter)

        self.grid_sensors = QGridLayout()
        self.sensors_radios = {}  # автоматически создаем чекбоксы из файла
        for index, sensor in enumerate(self.sensors):
            self.sensors_radios[sensor] = QRadioButton()
            self.sensors_radios[sensor].setText(sensor)
            self.grid_sensors.addWidget(self.sensors_radios[sensor], index // 4, index % 4,  1, 1)

        self.sensors_radios[self.sensors[0]].setChecked(True)
        self.tab_1_left_vert.addLayout(self.grid_sensors)

        self.tab_1_sensors_buttons = QHBoxLayout()

        self.button_sensors_run = QPushButton()
        self.button_sensors_run.setText("Запустить обработку изображения для данного сенсора")

        if state == 0:
            self.button_sensors_run.setEnabled(False)
        self.button_sensors_run.clicked.connect(self.run_sensors)
        self.tab_1_sensors_buttons.addWidget(self.button_sensors_run, alignment=QtCore.Qt.AlignLeft)

        self.button_sensors_change = QPushButton()
        self.button_sensors_change.setText("Открыть конфиг для изменения сенсоров")
        self.button_sensors_change.clicked.connect(self.select_sensors)
        self.tab_1_sensors_buttons.addWidget(self.button_sensors_change, alignment=QtCore.Qt.AlignRight)

        self.tab_1_left_vert.addLayout(self.tab_1_sensors_buttons)

        self.chose_guide = QLabel()
        self.chose_guide.setText(
        "При нажатии кнопки 'Выбрать область интереса на изображении' это окно закроется и откроется окно выбора области.\n"
        "На нем с зажатой левой кнопкой мышки нужно обвести требуюмую область.\n"
        "Если не получилось выделить, можно заново обвести облась - в окне справа будет показана выделенная область.\n"
        "Далее нужно закрыть окно выбора области и продолжить работу в этом окне.")
        self.tab_1_left_vert.addWidget(self.chose_guide, 0, QtCore.Qt.AlignHCenter)

        self.button_roi = QPushButton()
        self.button_roi.setText("Выбрать область интереса на изображении")
        if state == 0:
            self.button_roi.setEnabled(False)
        self.button_roi.clicked.connect(self.select_roi)
        self.tab_1_left_vert.addWidget(self.button_roi)

        self.button_deeplab = QPushButton()
        self.button_deeplab.setText("Поиск маски моделью DeeplabV3")
        if state == 0:
            self.button_deeplab.setEnabled(False)
        self.button_deeplab.clicked.connect(self.start_deeplab)
        self.tab_1_left_vert.addWidget(self.button_deeplab)

        self.button_refiner = QPushButton()
        self.button_refiner.setText("Уточнение границ")
        if state == 0:
            self.button_refiner.setEnabled(False)
        self.button_refiner.clicked.connect(self.start_refiner)
        self.tab_1_left_vert.addWidget(self.button_refiner)

        self.select_image_label = QLabel()
        self.select_image_label.setText(
        "Теперь выберите нужные вегетативные индексы на вкладке \"Индексы\".\n"
        "При необходимости можно загрузить свои нажатием кнопки сверху на вкладке \"Индексы\".\n"
        "Также выберите более подходящую маску из вычисленных.")
        self.tab_1_left_vert.addWidget(self.select_image_label)

        self.selector_image = QComboBox()
        self.selector_image.addItems(['Область изображения', 'Deeplab', 'С уточненными границами'])
        self.selector_image.currentIndexChanged.connect(self.on_selector_image_changed)
        if state == 0:
            self.selector_image.setEnabled(False)

        self.tab_1_left_vert.addWidget(self.selector_image)

        self.button_start = QPushButton()
        self.button_start.setText("Запустить вычисления индексов")
        if state == 0:
            self.button_start.setEnabled(False)
        self.button_start.clicked.connect(self.start)
        self.tab_1_left_vert.addWidget(self.button_start)

        self.empty_label = QLabel()
        self.empty_label.setText(" ")
        self.tab_1_left_vert.addWidget(self.empty_label)

        self.tab_1_horiz.addLayout(self.tab_1_left_vert, stretch=2)

        self.top_image = QLabel()
        self.tab_1_horiz.addWidget(self.top_image, stretch=3)

        self.tab_1.setLayout(self.tab_1_horiz)
        self.tabWidget.addTab(self.tab_1, "Изображение")  # В итоге все компилируем на 1 вкладке

        self.tab_2 = QWidget(self)  # Вторая вкладка

        self.tab_2_horez = QHBoxLayout()  # делим пополам вкладку

        self.tab_2_indeces_vert = QVBoxLayout()  # правый лейаут

        self.button_formulas_change = QPushButton()
        self.button_formulas_change.setText("Открыть конфиг для изменения формул")
        self.button_formulas_change.clicked.connect(self.select_indices)
        self.tab_2_indeces_vert.addWidget(self.button_formulas_change)

        self.label_6 = QLabel()  # лейбл индексы
        self.label_6.setText('Индексы')
        self.label_6.setFont(QtGui.QFont("Times", weight=QtGui.QFont.Bold))
        self.tab_2_indeces_vert.addWidget(self.label_6)

        self.label_12 = QLabel()  # лейбл составных индексов
        self.label_12.setText("Составные индексы")
        self.tab_2_indeces_vert.addWidget(self.label_12)

        self.formulas_dict = {
            'Complex indices': QGridLayout(),
            'Vegetation indices': QGridLayout(),
            'Water indices': QGridLayout(),
            'Geo indices': QGridLayout(),
            'Burn indices': QGridLayout()
        }
        self.checkboxes = {
            'Complex indices': {},
            'Vegetation indices': {},
            'Water indices': {},
            'Geo indices': {},
            'Burn indices': {}
        }

        for index, VI in enumerate(self.complex):
            self.checkboxes['Complex indices'][VI] = QCheckBox()
            self.checkboxes['Complex indices'][VI].setText(VI)
            self.formulas_dict['Complex indices'].addWidget(self.checkboxes['Complex indices'][VI], index // 4, index % 4,  1, 1)

        self.tab_2_indeces_vert.addLayout(self.formulas_dict['Complex indices'])

        self.label_7 = QLabel()  # лейбл индекса вегетации
        self.label_7.setText("Индексы вегитации")
        self.tab_2_indeces_vert.addWidget(self.label_7)

        for index, VI in enumerate(self.vegetation):
            self.checkboxes['Vegetation indices'][VI] = QCheckBox()
            self.checkboxes['Vegetation indices'][VI].setText(VI)
            self.formulas_dict['Vegetation indices'].addWidget(self.checkboxes['Vegetation indices'][VI], index // 4, index % 4,  1, 1)

        self.tab_2_indeces_vert.addLayout(self.formulas_dict['Vegetation indices'])

        self.label_8 = QLabel()  # лейбл водяных индексов
        self.label_8.setText("Водяные индексы")
        self.tab_2_indeces_vert.addWidget(self.label_8)

        self.checkboxes_water = {}

        for index, VI in enumerate(self.water):
            self.checkboxes['Water indices'][VI] = QCheckBox()
            self.checkboxes['Water indices'][VI].setText(VI)
            self.formulas_dict['Water indices'].addWidget(self.checkboxes['Water indices'][VI], index // 4, index % 4, 1, 1)

        self.tab_2_indeces_vert.addLayout(self.formulas_dict['Water indices'])

        self.label_9 = QLabel()  # лейбл геологически индексов
        self.label_9.setText("Геологические индексы")
        self.tab_2_indeces_vert.addWidget(self.label_9)

        for index, VI in enumerate(self.geo):
            self.checkboxes['Geo indices'][VI] = QCheckBox()
            self.checkboxes['Geo indices'][VI].setText(VI)
            self.formulas_dict['Geo indices'].addWidget(self.checkboxes['Geo indices'][VI], index // 4, index % 4,  1, 1)

        self.tab_2_indeces_vert.addLayout(self.formulas_dict['Geo indices'])

        self.label_10 = QLabel()  # лейбл индексов горения
        self.label_10.setText("Индексы горения")
        self.tab_2_indeces_vert.addWidget(self.label_10)

        for index, VI in enumerate(self.burn):
            self.checkboxes['Burn indices'][VI] = QCheckBox()
            self.checkboxes['Burn indices'][VI].setText(VI)
            self.formulas_dict['Burn indices'].addWidget(self.checkboxes['Burn indices'][VI], index // 4, index % 4,  1, 1)

        self.tab_2_indeces_vert.addLayout(self.formulas_dict['Burn indices'])

        self.empty_label_2 = QLabel()
        self.empty_label_2.setText(" ")
        self.tab_2_indeces_vert.addWidget(self.empty_label_2)

        self.tab_2_horez.addLayout(self.tab_2_indeces_vert)

        self.tab_2.setLayout(self.tab_2_horez)

        if state == 0:
            self.tab_2.setEnabled(False)

        self.tabWidget.addTab(self.tab_2, "Индексы")  # лейбл водяных индексов

        self.tabWidget.tabBarClicked.connect(self.tab_cliked)

        main_layout.addWidget(self.tabWidget)

    def open_file_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            'ims/2020/micasense',
            "Изображение (*.png *.jpg, *.tif)"
        )
        if filename:
            global filename_g
            filename_g = filename
            self.image_path = pathlib.Path(filename)
            self.filename_edit.setText(str(self.image_path))
            self.selector_resize.setEnabled(True)
            self.button_resize.setEnabled(True)


    def button_resize_image(self):

        self.scale_const = int(self.selector_resize.currentText())
        global scale_const_g
        scale_const_g = self.scale_const
        self.progress.setValue(5)
        print('Reading raw file')
        dataset = gdal.Open(pathlib.Path(filename_g), gdal.GA_ReadOnly)
        self.progress.setValue(10)
        image_datatype = dataset.GetRasterBand(1).DataType

        image = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount),
                         dtype=gdal_array.GDALTypeCodeToNumericTypeCode(image_datatype))

        self.progress.setValue(20)

        for b in range(dataset.RasterCount):
            self.progress.setValue(20 + (b + 1) * 5)
            band = dataset.GetRasterBand(b + 1)
            image[:, :, b] = band.ReadAsArray()

        image = image.astype(np.float32)

        print('Resizing')
        self.progress.setValue(75)
        image = cv2.resize(image, (int(dataset.RasterXSize / self.scale_const), int(dataset.RasterYSize / self.scale_const)))
        np.save('temp/image_array.npy', image)

        print('Resize done!')
        self.progress.setValue(100)
        self.button_sensors_run.setEnabled(True)

    def run_sensors(self):
        self.selected_sensor = self.sensors_radios[self.sensors[0]]
        global selected_sensor_g

        for sensor, radio in self.sensors_radios.items():
            if radio.isChecked():
                self.selected_sensor = sensor
                selected_sensor_g = sensor
                break

        self.colors_dict = eval(self.config_sensors.get('Sensors', self.selected_sensor))
        img = np.load('temp/image_array.npy')
        index = np.array([self.colors_dict['R'], self.colors_dict['G'], self.colors_dict['B']])
        img = img[:, :, index]

        min_val = np.min(img)
        max_val = np.max(img)

        for b in range(img.shape[2]):
            img[:, :, b] = (img[:, :, b] - min_val)* 1.0 / (max_val - min_val)

        self.rgb_image_array = (img * 255).astype(np.uint8)
        np.save('temp/rgb_array.npy', self.rgb_image_array)
        cv2.imwrite('temp/rgb.png', self.rgb_image_array)

        pixmap = QtGui.QPixmap('temp/rgb.png')
        pixmap = pixmap.scaled(self.top_image.size(), QtCore.Qt.KeepAspectRatio)

        global img_width, img_height
        img_width = pixmap.width()
        img_height = pixmap.height()

        self.top_image.setPixmap(pixmap)
        self.button_roi.setEnabled(True)
        self.tab_2.setEnabled(True)

    def sensors_again(self):

        image = np.load('temp/rgb_array.npy').astype(np.uint8)
        mask = np.load('temp/mask_roi.npy').astype(np.uint8)

        self.filename_edit.setText(filename_g)

        out = np.copy(image)
        out[(mask == 1)] = [0, 255, 0]
        out = cv2.addWeighted(out, 0.3, image, 0.7, 0, out)
        cv2.imwrite('temp/mask_roi.png', out)

        pixmap = QtGui.QPixmap('temp/mask_roi.png')
        pixmap = pixmap.scaled(self.top_image.size(), QtCore.Qt.KeepAspectRatio)

        global img_width, img_height
        img_width = pixmap.width()
        img_height = pixmap.height()

        self.top_image.setPixmap(pixmap)
        self.sensors_radios[selected_sensor_g].setChecked(True)
        self.selector_image.setEnabled(True)

    def select_sensors(self):
        os.startfile('Sensors.ini')

    def select_indices(self):
        os.startfile('Formulas.ini')

    def select_roi(self):
        self.button_deeplab.setEnabled(True)
        global roi_state
        roi_state = 0
        self.close()

    def start_deeplab(self):
        image = np.load('temp/rgb_array.npy').astype(np.float32)
        mask = np.load('temp/mask_roi.npy').astype(np.float32)

        x, y, _ = image.shape

        void_pixels = 1 - mask
        sample = {'image': image, 'gt': mask, 'void_pixels': void_pixels}

        trns = transforms.Compose([
            tr.CropFromMask(crop_elems=('image', 'gt', 'void_pixels'), relax=30, zero_pad=True),
            tr.FixedResize(
                resolutions={'gt': None, 'crop_image': (512, 512), 'crop_gt': (512, 512),
                             'crop_void_pixels': (512, 512)},
                flagvals={'gt': cv2.INTER_LINEAR, 'crop_image': cv2.INTER_LINEAR, 'crop_gt': cv2.INTER_LINEAR,
                          'crop_void_pixels': cv2.INTER_LINEAR}),
            tr.IOGPoints(sigma=10, elem='crop_gt', pad_pixel=10),
            tr.ToImage(norm_elem='IOG_points'),
            tr.ConcatInputs(elems=('crop_image', 'IOG_points')),
            tr.ToTensor()])

        tr_sample = trns(sample)

        inputs = tr_sample['concat'][None]
        inputs = inputs.to(device)
        outputs = net.forward(inputs)[-1]
        pred = np.transpose(outputs.cpu().data.numpy()[0, :, :, :], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)
        gt = tens2image(tr_sample['gt'])
        bbox = get_bbox(gt, pad=30, zero_pad=True)
        result = crop2fullmask(pred, bbox, gt, zero_pad=True, relax=0, mask_relax=False)

        light = np.zeros_like(image)
        light[:, :, 2] = 255.

        alpha = 0.5
        mask = (alpha * light + (1 - alpha) * image) * result[..., None]
        mask = cv2.cvtColor(np.uint8(mask), cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)[1]

        np.save('temp/mask_deeplab.npy', mask)
        # out = cv2.resize(out, (0, 0), fx=0.25, fy=0.25)
        out = np.copy(image)
        out[(mask == 255)] = [0, 255, 0]
        out = cv2.addWeighted(out, 0.3, image, 0.7, 0, out)
        cv2.imwrite('temp/mask_deeplab.png', out)

        pixmap = QtGui.QPixmap('temp/mask_deeplab.png')
        pixmap = pixmap.scaled(self.top_image.size(), QtCore.Qt.KeepAspectRatio)

        global img_width, img_height
        img_width = pixmap.width()
        img_height = pixmap.height()

        self.top_image.setPixmap(pixmap)

        self.button_refiner.setEnabled(True)


    def start_refiner(self):
        self.button_start.setEnabled(True)
        image = np.load('temp/rgb_array.npy').astype(np.uint8)
        mask = np.load('temp/mask_deeplab.npy')

        refined = refiner.refine(image, mask, fast=False, L=80)
        refined = np.where(refined > 200, 255, 0).astype(np.uint8)
        np.save('temp/mask_refined.npy', refined)

        out = np.copy(image)
        out[(refined == 255)] = [0, 255, 0]
        out = cv2.addWeighted(out, 0.3, image, 0.7, 0, out)
        cv2.imwrite('temp/mask_refined.png', out)
        # output = cv2.resize(output, (0, 0), fx=0.25, fy=0.25)
        pixmap = QtGui.QPixmap('temp/mask_refined.png')
        pixmap = pixmap.scaled(self.top_image.size(), QtCore.Qt.KeepAspectRatio)

        global img_width, img_height
        img_width = pixmap.width()
        img_height = pixmap.height()

        self.top_image.setPixmap(pixmap)

    def on_selector_image_changed(self):
        global img_width, img_height

        if self.selector_image.currentText() == 'Область изображения':
            pixmap = QtGui.QPixmap('temp/mask_roi.png').scaled(self.top_image.size(), QtCore.Qt.KeepAspectRatio)
            img_width = pixmap.width()
            img_height = pixmap.height()
            self.top_image.setPixmap(pixmap)

        elif self.selector_image.currentText() == 'Deeplab':
            pixmap = QtGui.QPixmap('temp/mask_deeplab.png').scaled(self.top_image.size(), QtCore.Qt.KeepAspectRatio)
            img_width = pixmap.width()
            img_height = pixmap.height()
            self.top_image.setPixmap(pixmap)

        elif self.selector_image.currentText() == 'С уточненными границами':
            pixmap = QtGui.QPixmap('temp/mask_refined.png').scaled(self.top_image.size(), QtCore.Qt.KeepAspectRatio)
            img_width = pixmap.width()
            img_height = pixmap.height()
            self.top_image.setPixmap(pixmap)

    def tab_cliked(self, index):
        if index == 0:
            for section, checkbox_dict in self.checkboxes.items():
                for VI, widget in checkbox_dict.items():
                    widget.setEnabled(True)
        #
        # if index == 1:
        #     for sensor, radio in self.sensors_radios.items():
        #         if radio.isChecked():
        #             self.selected_sensor = sensor
        #             break

            self.colors_dict = eval(self.config_sensors.get('Sensors', selected_sensor_g))
            keys = self.colors_dict.keys()

            for section, checkbox_dict in self.checkboxes.items():
                for VI, widget in checkbox_dict.items():
                    if 'RE' in self.config_formulas[section][VI] and 'RE' not in keys:
                        widget.setEnabled(False)
                    if 'NIR' in self.config_formulas[section][VI] and 'NIR' not in keys:
                        widget.setEnabled(False)
                    if 'MIR' in self.config_formulas[section][VI] and 'MIR' not in keys:
                        widget.setEnabled(False)
                    if 'TIR' in self.config_formulas[section][VI] and 'TIR' not in keys:
                        widget.setEnabled(False)

    def start(self):
        if self.selector_image.currentText() == 'Область изображения':
            mask_path = 'temp/mask_roi.npy'
        elif self.selector_image.currentText() == 'Deeplab':
            mask_path = 'temp/mask_deeplab.npy'
        else:
            mask_path = 'temp/mask_refined.npy'
        img = np.load('temp/image_array.npy')
        rgb_image = np.load('temp/rgb_array.npy')
        mask = np.load(mask_path)

        folder = f'indices/{filename_g.split('/')[-1][:-4]}'

        if not os.path.exists(folder):
            os.makedirs(folder)

        self.colors_dict = eval(self.config_sensors.get('Sensors', selected_sensor_g))

        for color, index in self.colors_dict.items():
            exec(f'{color}=img[..., index]')

        for section, checkbox_dict in self.checkboxes.items():
            for VI, widget in checkbox_dict.items():
                if widget.isChecked():

                    print(self.config_formulas[section][VI])
                    VI_arr = eval(self.config_formulas[section][VI]) #.astype(np.uint8)
                    VI_arr[np.isnan(VI_arr)] = 0
                    cmap = plt.cm.RdYlGn
                    norm = plt.Normalize(vmin=VI_arr.min(), vmax=VI_arr.max())
                    image = cmap(norm(VI_arr))
                    image = np.delete(image, 3, 2)
                    image = np.uint8(image * 255)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    out = np.copy(rgb_image)
                    out[(mask != 0)] = image[(mask != 0)]
                    zxc = cv2.imwrite(f'{folder}/{VI}.png', out)

        if selected_sensor_g != 'rgb':
            VI_arr = {}
            for index, formula in self.config_formulas.items('Final indices'):
                temp = eval(formula)
                temp[mask == 0] = np.nan
                if index == 'sipi2':
                    temp[temp < 0.5] = np.nan
                    temp[temp > 2] = np.nan
                temp = (temp - np.nanmin(temp)) / (np.nanmax(temp) - np.nanmin(temp))
                temp[np.isnan(temp)] = 0
                VI_arr[index] = temp
            veg = (VI_arr['mcari'] + VI_arr['osavi'] + VI_arr['mnli'] + VI_arr['tvi'] + VI_arr['gndvi']) / 5
            stress = (VI_arr['sipi2'] + VI_arr['mari']) / 2
            result = (veg + 1 - stress) / 2
            np.save('temp/veg.npy', veg)
            np.save('temp/stress.npy', stress)
            np.save('temp/result.npy', result)

            cmap = plt.cm.RdYlGn
            norm = plt.Normalize(vmin=result.min(), vmax=result.max())
            result = cmap(norm(result))
            result = np.delete(result, 3, 2)
            result = np.uint8(result * 255)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            out = np.copy(rgb_image)
            out[(mask != 0)] = result[(mask != 0)]

            zxc = cv2.imwrite(f'{folder}/result.png', out)
            pixmap = QtGui.QPixmap(f'{folder}/result.png').scaled(self.top_image.size(), QtCore.Qt.KeepAspectRatio)
            img_width = pixmap.width()
            img_height = pixmap.height()
            self.top_image.setPixmap(pixmap)

    def contextMenuEvent(self, e):
        if selected_sensor_g != 'rgb':
            if self.selector_image.currentText() == 'Область изображения':
                mask = 'temp/mask_roi.npy'
            elif self.selector_image.currentText() == 'Deeplab':
                mask = 'temp/mask_deeplab.npy'
            else:
                mask = 'temp/mask_refined.npy'
            mask = np.load(mask)
            context = QMenu(self)
            click_pos = e.pos()
            veg = np.load('temp/veg.npy')
            stress = np.load('temp/stress.npy')
            result = np.load('temp/result.npy')

            pixmap_pos = self.top_image.pos()
            image_x = pixmap_pos.x() + 13
            image_y = pixmap_pos.y() + 40
            if image_x <= click_pos.x() <= image_x + img_width and image_y <= click_pos.y() <= image_y + img_height:
                rel_pos_x = click_pos.x() - image_x
                rel_pos_y = click_pos.y() - image_y
                rescaled_x = rel_pos_x * veg.shape[1] // img_width
                rescaled_y = rel_pos_y * veg.shape[0] // img_height
                if mask[rescaled_y, rescaled_x] != 0:
                    act_1 = QAction(f"Result: {result[rescaled_y, rescaled_x]}", self)
                    act_1.setEnabled(False)
                    context.addAction(act_1)

                    act_2 = QAction(f"Vegetation: {veg[rescaled_y, rescaled_x]}", self)
                    act_2.setEnabled(False)
                    context.addAction(act_2)

                    act_3 = QAction(f"Stress: {stress[rescaled_y, rescaled_x]}", self)
                    act_3.setEnabled(False)
                    context.addAction(act_3)

                    context.exec(e.globalPos())

if __name__ == '__main__':
    # Глобальные переменные - константы
    roi_state = 0
    state = 1
    filename_g = 'ims/2020/micasense/2020.06.09_Micasense_Olgino.tif'
    scale_const_g = 1
    selected_sensor_g = 'micasense'
    img_width = 0
    img_height = 0

    # задаем сетку
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('Using GPU')
        # refiner = refine.Refiner(device='cuda:0')
    else:
        raise ValueError('Use GPU')

    # deeplab v3
    net = Network(nInputChannels=5,
                  num_classes=1,
                  backbone='resnet101',
                  output_stride=16,
                  sync_bn=None,
                  freeze_bn=False)

    pretrain_dict = torch.load('ims/IOG_PASCAL.pth')
    net.load_state_dict(pretrain_dict)
    net.to(device)
    net.eval()

    refiner = refine.Refiner(device="cuda:0")

    while True:
        if roi_state == 0:
            roi_state = 1
            app = QApplication([])
            app.setStyle('Fusion')
            app.setStyleSheet("QWidget{font-size: 16pt;}}")
            window = MainWindow()
            app.exec()
            if roi_state == 0:
                state = 1
                chose()
        else:
            exit(0)
