from Design_py import untitled
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5 import (QtWidgets)
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import sympy
import functools
from random import uniform

class MainWindow(QtWidgets.QWidget, untitled.Ui_Form):
    def __init__(self, parent=None):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. файле design.ру
        super().__init__(parent)
        self.setupUi(self) # Это нужно для инициализации нашего дизайна
        self.setFixedSize(self.width(), self .height())
        self.average=[]
        self.probability=[]
        self.pushButton.clicked.connect(self.part1_button)
        self.pushButton_2.clicked.connect(self.part2_button)
        self.graf.clicked.connect(self.graf_f)
        self.graf.setEnabled(False)
        self.graf_2.clicked.connect(self.graf_future)
        self.graf_2.setEnabled(False)
        self.future.clicked.connect(self.future_next)

        self.average = []
        self.probability = []
        self.average_for_similar_n = []
        self.probability_for_similar_n = []

    def Puass(self, lambd, k):
        return ((np.e ** (-lambd)) * (lambd ** k) / np.math.factorial(k))

    def gen_intervals(self, n, h, p, M, samples):
        l = 0
        eps = (1 / n) * 5 #погрешность
        p_now = 0
        M_left = 0
        M_right = 0
        while abs(p - p_now) > eps:
            current = l - h
            l = l + h
            while (p - p_now) < 0:
                l = current
                h = h / 2
                l = l + h
                M_left = M - l / 9
                M_right = M + (5 * l / 2)
                t = 0
                for i in samples:
                    if M_left <= i < M_right:
                        t = t + 1
                p_now = t / len(samples)
            M_left = M - l / 9
            M_right = M + (5 * l / 2)
            t = 0
            for i in samples:
                if M_left <= i < M_right:
                    t = t + 1
            p_now = t / len(samples)
            # print('p_now1 =', p_now)
            # print('l_gen1 =', l_gen)

        return M_left, M_right,l

    def M_and_D(self, samples):
        varion = {}
        M = 0
        D = 0
        for i in samples:
            if i in varion:
                varion[i] = varion[i] + 1
            else:
                varion[i] = 1

        for i in varion:
            varion[i] = varion[i] / len(samples)

        varion = sorted(varion.items())

        for i in range(0, len(varion)):
            M = M + varion[i][0] * varion[i][1]

        sample_variance = 0
        for i in range(0, len(varion)):
            D = D + (((varion[i][0] - M) ** 2) * varion[i][1])
        sample_variance = (len(varion) * D) / (len(varion) - 1)
        return M, D, sample_variance

    def var_row(self, n, result_arr):
        l = (result_arr[-1] - result_arr[0]) / n
        arr = []
        arr_in_interval = []

        for i in range(0, n):
            for j in range(0, len(result_arr)):
                if result_arr[0] + l * (i) <= result_arr[j] < (result_arr[0] + l * (i + 1)):
                    arr_in_interval.append(result_arr[j])
            arr.append(arr_in_interval)
            arr_in_interval = []
        col_el_in_interval = [] #количество элементов в интервале
        probability = [] #вероятность
        for i in range(0, n):
            col_el_in_interval.append(len(arr[i]))
            if i != 0:
                probability.append(probability[i - 1] + (len(arr[i]) / len(result_arr)))
            else:
                probability.append(len(arr[i]) / len(result_arr))
        average = []
        for i in range(0, n):
            interval1 = result_arr[0] + l * i
            interval2 = result_arr[0] + l * (i + 1)
            average.append([interval1, interval2])
        return probability,average, col_el_in_interval, l

    def get_EURRUB(self, file_name):
        eurrub_arr = []
        file = open(file_name, 'r', encoding='utf-8')

        for line in file.readlines():
            if line.strip():
                if line.split()[1] != '<VOL>':
                    eurrub_arr = eurrub_arr + [line.split()[1]]
        file.close()
        return sorted(list(map(float, eurrub_arr)))

    def part1_button(self):
        try:
            n = int(self.number.text())
        except:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Введите целое число!")
            msgBox.exec()
            return ()
        if n < 10:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Введите число больше 1!")
            msgBox.exec()
            return ()
        try:
            A = float(self.A.text())
        except:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Введите число больше 0!")
            msgBox.exec()
            return ()
        if A<0:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Введите число больше 0!")
            msgBox.exec()
            return ()
        try:
            s = int(self.s.text())
        except:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Введите целое число больше 0!")
            msgBox.exec()
            return ()
        if s < 0:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Введите число больше 0!")
            msgBox.exec()
            return ()
        try:
            lambd = float(self.lambd.text())
        except:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Введите число!")
            msgBox.exec()
            return ()
        if lambd <= 0:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Введите число больше 0!")
            msgBox.exec()
            return ()

        h = 2 / n #шаг

        x_array = np.arange(0, 2, h) #диапазон по графику
        y_I = np.zeros(len(x_array))
        y_F = np.zeros(len(x_array))

        #1.
        #Заданное распределение
        func_samples = np.zeros(n) #массив для значений случайной величины

        x = Symbol('x') #не переменная, а объект символа
        g = (5 * (A ** (s + 1)) / (sympy.factorial(s))) #гамма
        I = integrate(
            g * (x ** (5 * s + 4)) * sympy.E ** (-A * (x ** 5)), (x, 0, x)) #от 0 до x чтобы вывести формулу

        recursion = (1 - sympy.E ** (-A * (x ** 5))) / A #последний элемент рекурсия
        for i in range(s - 1, 0, -1):
            recursion = (-(s - i + 1) / A) * (sympy.E ** (-A * (x ** 5)) * (x ** (5 * (s - i))) - recursion) #рекурсия

        F = (-g / (5 * A)) * ((x ** (5 * s)) * (sympy.E ** (-A * (x ** 5))) - recursion) #рекуррентная формула

        for i in range(0, len(x_array)): #таблица значений функции по формуле распределения
            y_I[i] = I.evalf(subs={x: x_array[i]})
            y_F[i] = F.evalf(subs={x: x_array[i]})

        xy_dictionary = dict(zip(y_F, x_array))#ключ - вероятность(y), значение - случайная величина(x)

        y_rand = np.random.random_sample(n)#от 0 до 1 (не включая 1) равномерно распределенные

        for i in range(0, n):
            func_samples[i] = xy_dictionary[
                min(y_F, key=lambda a: abs(a - y_rand[i]))]  # полученная выборка случайных величин

        self.given_distribution.clear()
        self.given_distribution.setColumnCount(1)
        self.given_distribution.setRowCount(0)
        self.given_distribution.setHorizontalHeaderLabels(["Значения"])
        for i in range(0, n):
            self.given_distribution.insertRow(self.given_distribution.rowCount())
            self.given_distribution.setItem(self.given_distribution.rowCount() - 1, 0, QTableWidgetItem(str(f'{func_samples[i]:.3f}')))

        # Пуассоновское распределение
        func_samples_Puass = np.zeros(n)

        k = np.arange(0, 15, 1) #диапазон по графику
        p_Puass = np.zeros(len(k))
        p_Puass[0] = ((sympy.E ** (-lambd)) * (lambd ** k[0])) / (np.math.factorial(k[0]))
        for i in range(1, len(k)):
            p_Puass[i] = p_Puass[i - 1] + ((sympy.E ** (-lambd)) * (lambd ** k[i])) / (np.math.factorial(k[i]))
        pk_dictionary = dict(zip(p_Puass, k))

        for i in range(0, n):
            func_samples_Puass[i] = pk_dictionary[min(p_Puass, key=lambda a: abs(a - y_rand[i]))]

        self.Poisson_distribution.clear()
        self.Poisson_distribution.setColumnCount(1)
        self.Poisson_distribution.setRowCount(0)
        self.Poisson_distribution.setHorizontalHeaderLabels(["Значения"])
        for i in range(0, n):
            self.Poisson_distribution.insertRow(self.Poisson_distribution.rowCount())
            self.Poisson_distribution.setItem(self.Poisson_distribution.rowCount() - 1, 0,
                                            QTableWidgetItem(str(f'{func_samples_Puass[i]:.3f}')))

        #2.
        #Заданное распределение
        given_M, given_D, given_sample_variance = self.M_and_D(func_samples)

        self.given_M.clear()
        self.given_M.setText(str(f'{given_M:.5f}'))

        self.given_D.clear()
        self.given_D.setText(str(f'{given_sample_variance:.5f}'))

        # Пуассоновское распределение
        Poisson_M, Poisson_D, Poisson_sample_variance = self.M_and_D(func_samples_Puass)

        self.Poisson_M.clear()
        self.Poisson_M.setText(str(f'{Poisson_M:.5f}'))

        self.Poisson_D.clear()
        self.Poisson_D.setText(str(f'{Poisson_sample_variance:.5f}'))

        #3.
        #Заданное распределение
        sd = symbols('s')
        expr_diff = Derivative(sympy.log(sympy.factorial(sd)), sd).doit()

        Summ = sum([*map(lambda x: x ** 5, func_samples)])
        Multi = functools.reduce(lambda a, b: a * b, func_samples)

        s_param = 0
        I_right = len(func_samples) * np.log(len(func_samples)) - len(func_samples) * np.log(
            Summ) - (5 * np.log(Multi))
        s_prev = 0
        s_after = 0
        temp_prev = expr_diff.evalf(subs={sd: 0}) - len(func_samples) * np.log(0 + 1)
        while True:
            s_param = s_param + 1
            temp = expr_diff.evalf(subs={sd: s_param}) - len(func_samples) * np.log(s_param + 1)

            if (temp_prev >= I_right and temp >= I_right) or (
                    temp_prev < I_right and temp < I_right):
                s_prev = temp
            else:
                s_after = temp
                break
        if abs(s_prev - I_right) <= abs(s_after - I_right):
            s_param = s_param - 1

        A_param = (len(func_samples) * (s_param + 1)) / Summ

        self.estimation_s.clear()
        self.estimation_s.setText(str(s_param))

        self.estimation_A.clear()
        self.estimation_A.setText(str(f'{A_param:.5f}'))

        # Пуассоновское распределение
        self.estimation_lambd.clear()
        self.estimation_lambd.setText(str(f'{Poisson_M:.5f}'))

        #4.
        #Заданное распределение
        p = 0.9
        try:
            p = float(self.s.text())
        except:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Введите  число больше 0 и меньше 1!")
            msgBox.exec()
            return ()
        if p<0 or p>1:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Введите число больше 0 и меньше 1!")
            msgBox.exec()
            return ()
        l = Symbol('l')
        M = integrate(
            g * (x ** (5 * s + 4)) * x * sympy.E ** (
                        -A * (x ** 5)), (x, 0, oo)).evalf()
        print(M)

        I = integrate(
            g * (x ** (5 * s + 4)) * sympy.E ** (
                        -A * (x ** 5)), (x, M - (l / 9), M + (5 * l / 2)))
        print(I)

        # Пуассоновское распределение
        M_Puass = lambd
        Sum_Puass = 0
        l_Puass = 0
        while Sum_Puass < p:
            a_Puass = round(M_Puass - (l_Puass / 9)) #т.к. дискретное распределение
            b_Puass = round(M_Puass + ((5 * l_Puass) / 2))
            Sum_Puass = 0
            if a_Puass < 0:
                a_Puass = 0
            for i in range(a_Puass, b_Puass):
                Sum_Puass = Sum_Puass + self.Puass(lambd, i)
            l_Puass = l_Puass + 0.1

        self.Poisson_left.clear()
        self.Poisson_left.setText(str(round(M_Puass - (l_Puass / 9))))

        self.Poisson_right.clear()
        self.Poisson_right.setText(str(round(M_Puass + ((5 * l_Puass) / 2))))

        self.Poisson_l.clear()
        self.Poisson_l.setText(str(f'{l_Puass:.3f}'))

        #5.
        #Заданное распределение
        M_gen_given_data_left, M_gen_given_data_right, l_gen_given_data = self.gen_intervals(n,h,p,given_M,func_samples)

        self.given_data_left.clear()
        self.given_data_left.setText(str(f'{M_gen_given_data_left:.3f}'))

        self.given_data_right.clear()
        self.given_data_right.setText(str(f'{M_gen_given_data_right:.3f}'))

        self.given_data_l.clear()
        self.given_data_l.setText(str(f'{l_gen_given_data:.3f}'))

        # Пуассоновское распределение
        M_gen_Poisson_data_left, M_gen_Poisson_data_right, l_gen_Poisson_data = self.gen_intervals(n,h,p,Poisson_M,func_samples_Puass)

        self.Poisson_data_left.clear()
        self.Poisson_data_left.setText(str(f'{M_gen_Poisson_data_left:.3f}'))

        self.Poisso_data_right.clear()
        self.Poisso_data_right.setText(str(f'{M_gen_Poisson_data_right:.3f}'))

        self.Poisson_data_l.clear()
        self.Poisson_data_l.setText(str(f'{l_gen_Poisson_data:.3f}'))

    def part2_button(self):
        try:
            n = int(self.number_2.text())
        except:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Введите целое число!")
            msgBox.exec()
            return ()
        if n<2 or n>33:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Введите число больше 2 и меньше 33!")
            msgBox.exec()
            return ()

        result_arr = self.get_EURRUB('EURCB_161209_211210.txt')

        #1.
        self.probability, self.average, col_el_in_interval, l = self.var_row(n, result_arr)

        self.Variation_series.clear()
        self.Variation_series.setColumnCount(2)
        self.Variation_series.setRowCount(n)
        self.Variation_series.setHorizontalHeaderLabels(["Количество элементов","Интервал"])
        for i in range(0, n):
            rrr = (
                        "от " + f'{result_arr[0] + l * (i):.5f}' + "до " + f'{(result_arr[0] + l * (i + 1)):.5f}')
            print(str(rrr), '=', str(col_el_in_interval[i]))
            self.Variation_series.setItem(i, 1, QTableWidgetItem(str(rrr)))
            self.Variation_series.setItem(i - 1, 2, QTableWidgetItem(str(col_el_in_interval[i])))

        self.graf.setEnabled(True)

        # 3.

        M, D, sample_variance = self.M_and_D(result_arr)
        Mx2 = D + M ** 2

        a = 20
        b = -40 * M
        c = -5 * Mx2 + 25 * M ** 2 + 5 * M
        Discr = b ** 2 - 4 * a * c
        print('discr= ', Discr)

        lamb21 = (-b + np.sqrt(Discr)) / (2 * a)
        lamb22 = (-b - np.sqrt(Discr)) / (2 * a)
        print('lamb21= ', lamb21)
        print('lamb22= ', lamb22)
        lamb11 = 5 * M - 4 * lamb21
        lamb12 = 5 * M - 4 * lamb22
        print('my resh = ', [(lamb21, lamb11), (lamb22, lamb12)])
        print('M = ', M)
        print('D = ', D)

        self.p1_calculatad.clear()
        self.p1_calculatad.setText(str(f'{lamb11:.5f}'))

        self.p2_calculatad.clear()
        self.p2_calculatad.setText(str(f'{lamb21:.5f}'))

        #4.
        #M, D, sample_variance = self.M_and_D(result_arr)

        g = float(self.gamma_Student.currentText())
        g = g / 2
        file = open('laplas.txt', 'r')

        result_laplas_arr = []
        while True:
            line = file.readline()
            if not line:
                break
            result_laplas_arr.append([*map(float, line.strip('\n').split(';'))])
        file.close()
        t = min(result_laplas_arr, key=lambda a: abs(a[1] - g))[0]
        eps = t * np.sqrt(D)
        x_aver = sum(result_arr) / len(result_arr)

        self.D_left.clear()
        self.D_left.setText(str(f'{x_aver - eps:.5f}'))

        self.D_right.clear()
        self.D_right.setText(str(f'{x_aver + eps:.5f}'))

        # 5.
        file = open('Student.txt', 'r')
        result_Student_arr = []
        while True:
            line = file.readline()
            if not line:
                break
            result_Student_arr.append([*map(float, line.strip('\n').split(' '))])
        file.close()
        Student_significance_level = {0.01: 1, 0.05: 2, 0.1: 3, 0.15: 4, 0.2: 5, 0.25: 6, 0.3: 7}
        if (n - 1) > 120:
            k = 120
        else:
            k = n - 1
        t_g = result_Student_arr[k - 1][Student_significance_level[g]]
        eps = (t_g * np.sqrt(D)) / np.sqrt(k)

        self.Student_left.clear()
        self.Student_left.setText(str(f'{x_aver - eps:.5f}'))

        self.Student_right.clear()
        self.Student_right.setText(str(f'{x_aver + eps:.5f}'))

        # 6.
        g = float(self.gamma_Hi.currentText())
        g = g / 2
        l = (result_arr[-1] - result_arr[0]) / n
        arr = []
        arr_in_interval = []
        interval_border = []

        for i in range(0, n):
            for j in range(0, len(result_arr)):
                if result_arr[0] + l * (i) <= result_arr[j] < (result_arr[0] + l * (i + 1)):
                    arr_in_interval.append(result_arr[j])
            interval_border.append([result_arr[0] + l * (i), result_arr[0] + l * (i + 1)])
            arr.append(arr_in_interval)
            arr_in_interval = []
        col_el_in_interval = []
        for i in range(0, n):
            col_el_in_interval.append(len(arr[i]))

        inter_check = False
        while not inter_check:
            inter_check = True
            for i in range(0, len(col_el_in_interval)):
                if col_el_in_interval[i] < 5:
                    inter_check = False
                    if len(col_el_in_interval) <= 2:
                        print('ошибка количество интервалов меньше 2')
                        inter_check = True
                        break
                    if i == 0:
                        interval_border[0][1] = interval_border[1][1]
                        col_el_in_interval[0] += col_el_in_interval[1]
                        interval_border.pop(1)
                        col_el_in_interval.pop(1)
                    else:
                        interval_border[i][0] = interval_border[i - 1][0]
                        col_el_in_interval[i] += col_el_in_interval[i - 1]
                        interval_border.pop(i - 1)
                        col_el_in_interval.pop(i - 1)
                    break

        average = []
        for i in range(0, len(interval_border)):
            average.append((interval_border[i][0] + interval_border[i][1]) / 2)

        a = 0  #выборочное среднее
        for i in range(0, len(average)):
            a = a + (average[i] * (col_el_in_interval[i]))
        a = a / len(result_arr)

        D = 0
        for i in range(0, len(average)):
            D = D + ((average[i] - a) ** 2) * col_el_in_interval[i]
        D = D / len(result_arr)

        sigma = np.sqrt(D)

        p = []
        F2 = (interval_border[0][1] - a) / sigma
        t2 = min(result_laplas_arr, key=lambda a: abs(a[0] - abs(F2)))[1]
        if F2 < 0:
            t2 = -t2
        p.append((t2 + 1 / 2) * len(result_arr))
        for i in range(1, len(interval_border) - 1):

            F1 = (interval_border[i][0] - a) / sigma
            F2 = (interval_border[i][1] - a) / sigma
            t1 = min(result_laplas_arr, key=lambda a: abs(a[0] - abs(F1)))[1]
            t2 = min(result_laplas_arr, key=lambda a: abs(a[0] - abs(F2)))[1]
            if F1 < 0:
                t1 = -t1
            if F2 < 0:
                t2 = -t2
            p.append((t2 - t1) * len(result_arr))

        F1 = (interval_border[-1][0] - a) / sigma
        t1 = min(result_laplas_arr, key=lambda a: abs(a[0] - abs(F1)))[1]
        if F1 < 0:
            t1 = -t1
        p.append((1 / 2 - t1) * len(result_arr))

        hi = 0
        for i in range(0, len(interval_border)):
            hi = hi + (col_el_in_interval[i] ** 2) / p[i]
        hi = hi - len(interval_border)

        k = len(col_el_in_interval) - 2 - 1 #k=m-r-1, где m - количество интервалов, в r - количество параметров распределения = 2

        file = open('hi.txt', 'r')
        result_Hi_arr = []
        while True:
            line = file.readline()
            if not line:
                break
            result_Hi_arr.append([*map(float, line.strip('\n').split(' '))])
        file.close()
        Hi_significance_level = {0.01: 1, 0.025: 2, 0.05: 3, 0.95: 4, 0.975: 5, 0.99: 6}

        hi_kvant = result_Hi_arr[k - 1][Hi_significance_level[g]]

        self.hi_in_table.clear()
        self.hi_in_table.setText(str(hi_kvant))

        self.ih_2.clear()
        self.ih_2.setText(str(f'{hi:.5f}'))

        if hi <= hi_kvant:
            self.result.clear()
            self.result.setText(str('Ho - принимается'))
        else:
            self.result.clear()
            self.result.setText(str('Ho - не принимается'))

    def future_next(self):
        #7.
        p_ver = 0.6
        try:
            p_ver = float(self.p_future.text())
        except:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Введите число!")
            msgBox.exec()
            return ()
        if p_ver <= 0 or p_ver >= 1:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Введите число больше 0 и меньше 1!")
            msgBox.exec()
            return ()

        result_arr = self.get_EURRUB('EURCB_211113_211210.txt')

        M, D, _ = self.M_and_D(result_arr)
        l = np.sqrt(D / (1 - p_ver))

        fluctuation = uniform(-l, l)
        future_value = M + fluctuation

        self.future_meaning.clear()
        self.future_meaning.setText(str(f'{future_value:.5f}'))

        self.range.clear()
        text_for_field ='от ' + str(f'{M - l:.5f}') + ' до ' + str(f'{M + l:.5f}')
        self.range.setText(str(text_for_field))
        self.graf_2.setEnabled(True)


    def graf_f(self):
        plt.close()
        plt.figure(figsize=(4,4))

        if self.probability != []:
            self.average_for_similar_n = self.average
            self.probability_for_similar_n = self.probability
            for i in range(0,len(self.probability)):
                plt.plot(self.average[i],[self.probability[i]]*2,'-r')
        else:
            for i in range(0,len(self.probability_for_similar_n)):
                plt.plot(self.average_for_similar_n[i],[self.probability_for_similar_n[i]]*2,'-r')
        self.average = []
        self.probability = []
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('P')
        plt.title('Статистическая функция распределения')

        plt.show()

    def graf_future(self):
        plt.figure()
        x_date = []
        y_value = []

        file = open('EURCB_211113_211210.txt', 'r', encoding='utf-8')
        for line in file.readlines():
            if line.strip():
                if line.split()[1] != '<VOL>':
                    temp = line.split()
                    x_date = x_date + [temp[0][6:]+'.'+temp[0][4:6]]
                    y_value = y_value + [temp[1]]
        file.close()
        y_value = list(map(float, y_value))

        x_date = x_date + ['11.12']
        y_value = y_value + [float(self.future_meaning.text())]

        plt.plot(x_date, y_value, label='Курс рубля')
        plt.plot(x_date[-1], y_value[-1], 'ro')
        plt.annotate('Будущее значение', xy=(x_date[-4], y_value[-1]+0.1), xytext=(x_date[-4], y_value[-1]+0.1))

        plt.xlabel('Дата')
        plt.ylabel('Курс')
        plt.title('Курс рубля 13.11 - 11.12')

        plt.show()


