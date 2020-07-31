from Utils.db_utils import *
from Utils.utils import *
from kivy.config import Config
Config.set('graphics', 'window_state', 'maximized')

from kivy.core.window import Window
Window.clearcolor = (1, 1, 1, 1)

import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.image import Image

from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture

from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.garden.graph import Graph, MeshLinePlot
from kivy.graphics import Color, Rectangle

import sys
import datetime
import cv2

kivy.require("1.11.1")


global usernames
global embeddings

global attendance
global camera_index
global show_attendance_obj
global show_visual_obj

camera_index = 0

attendance = dict()

class Day_Select(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_path = 'Output/'
        self.cols = 2
        self.flag = 0
        self.padding = [100, 100, 100, 100]
        self.spacing = [20, 20]
        self.name = TextInput(multiline = False, size_hint = (.2, None), height = 40)
        add = Button(text='ADD', font_size = 30, italic = True, background_color = [255, 1, 1, 1])
        #Next = Button(text='NEXT', font_size = 30, italic = True, background_color = [1, 1, 255, 1])
        self.label = Label(text='Add Day!!!', font_size = 38, color = [255, 255, 255, 1])


        add.bind(on_press = self.add)
        #Next.bind(on_press = self.gonext)

        self.button_layout = GridLayout(rows = 4, spacing = [20, 20])

        self.button_layout.add_widget(self.name)
        self.button_layout.add_widget(add)
        #self.button_layout.add_widget(Next)
        self.button_layout.add_widget(self.label)
        self.add_widget(self.button_layout)


    def add(self, instance):
        if len(self.name.text) != 0:
            global day
            day = self.name.text
            UI_interface.screen_manager.current = "Home"
            if day == 'day_1':
                print("First day")
                embeddings, usernames = readAllBlobData()
                for username in usernames:
                    if username not in attendance.keys():
                        attendance[username] = 'Absent'
                        insert_data(username, 0, day)
            else:
                print("Not 1st day")
                bar_graph(day)
                pie_chart(day)
                embeddings, usernames = readAllBlobData()
                for username in usernames:
                    if username not in attendance.keys():
                        attendance[username] = 'Absent'
                        insert_data(username, 0, day)

        else:
            self.button_layout.remove_widget(self.label)
            self.label = Label(text = 'Enter Day', font_size = 38, color = [255, 255, 255, 1])
            self.button_layout.add_widget(self.label)

class Home_Page(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 2
        self.padding = [100, 100, 100, 100]
        self.add_widget(Image(source = 'Attendance.jpg'))

        buttons = GridLayout(cols = 2)

        buttons.spacing = [20, 20]

        button_1 = Button(text='TAKE \nATTENDANCE', font_size = 30, italic = True, background_color = [1, 255, 1, 1])
        button_2 = Button(text='SHOW \nATTENDANCE', font_size = 30, italic = True, background_color = [255, 1, 1, 1])
        button_3 = Button(text='ADD \nSTUDENT', font_size = 30, italic = True, background_color = [1, 1, 255, 1])
        button_4 = Button(text='REMOVE \nSTUDENT', font_size = 30, italic = True, background_color = [100, 140, 0, 1])

        button_1.bind(on_press = self.take_attendance)
        button_2.bind(on_press = self.show_attendance)
        button_3.bind(on_press = self.add_student)
        button_4.bind(on_press = self.remove_student)

        buttons.add_widget(button_1)
        buttons.add_widget(button_2)
        buttons.add_widget(button_3)
        buttons.add_widget(button_4)

        self.add_widget(buttons)


    def take_attendance(self, instance):
        global usernames, embeddings
        embeddings, usernames = readAllBlobData()
        UI_interface.screen_manager.current = "Attendance"

    def show_attendance(self, instance):
        global usernames, embeddings
        global show_attendance_obj
        embeddings, usernames = readAllBlobData()

        show_attendance_obj.show()
        UI_interface.screen_manager.current = "Show_Attendance"

    def add_student(self, instance):
        UI_interface.screen_manager.current = "Add_Student"

    def remove_student(self, instance):
        global remove_student_obj
        remove_student_obj.show()
        UI_interface.screen_manager.current = "Remove_Student"


class Attendance_Page(GridLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_path = 'Output/'
        self.flag = 0
        self.cols = 2
        self.padding = [100, 100, 100, 100]
        self.spacing = [20, 20]
        self.begin = Button(text='BEGIN', font_size = 30, italic = True, background_color = [1, 255, 1, 1])
        self.begin.bind(on_press = self.start)
        self.add_widget(self.begin)
        self.back = Button(text='GO BACK', font_size = 30, italic = True, background_color = [255, 1, 1, 1])
        self.back.bind(on_press = self.goback)
        self.add_widget(self.back)


    def start(self, instance):
        global camera_index
        self.flag = 1
        self.img=Image()
        self.layout = BoxLayout()
        self.layout.add_widget(self.img)
        self.remove_widget(self.begin)
        self.remove_widget(self.back)
        self.add_widget(self.layout)

        mark = Button(text='MARK ME', font_size = 30, italic = True, background_color = [255, 1, 1, 1])
        back = Button(text='GO BACK', font_size = 30, italic = True, background_color = [1, 1, 255, 1])
        self.label = Label(text='Mark Your Attendance!!', font_size = 38, color = [255, 255, 255, 1])

        mark.bind(on_press = self.recognize)
        back.bind(on_press = self.goback)

        self.button_layout = GridLayout(rows = 3, spacing = [20, 20])

        self.button_layout.add_widget(mark)
        self.button_layout.add_widget(back)
        self.button_layout.add_widget(self.label)

        self.add_widget(self.button_layout)

        self.capture = cv2.VideoCapture(camera_index)
        self.event = Clock.schedule_interval(self.update, 1.0/33.0)


    def update(self, instance):
        _, self.frame = self.capture.read()
        self.frame = extract_all_faces(self.frame)
        buf1 = cv2.flip(self.frame, 0)
        buf = buf1.tostring()
        texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img.texture = texture


    def recognize(self, instance):
        global day
        ts = datetime.datetime.now()
        img_name = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        img_path = self.output_path + img_name
        cv2.imwrite(img_path, self.frame)
        print("[INFO] saved {}".format(img_name))
        embedding, flag = generate_embedding(img_path)

        if embeddings is not None:
            if flag == 1:
                ones_matrix = np.ones((len(usernames), 1))
                embedding_matrix = np.matmul(ones_matrix, embedding.detach().numpy())
                distances = calc_distance(embedding_matrix, embeddings)
                if (distances[np.argmin(distances)] < 1.0000):
                    print(usernames[np.argmin(distances)] + ' Marked')
                    self.button_layout.remove_widget(self.label)
                    self.label = Label(text=usernames[np.argmin(distances)] + ' Marked', font_size = 38, color = [255, 255, 255, 1])
                    self.button_layout.add_widget(self.label)
                    attendance[usernames[np.argmin(distances)]] = "Present"
                    insert_data(usernames[np.argmin(distances)],1,day)
                else:
                    self.button_layout.remove_widget(self.label)
                    self.label = Label(text = 'User Not Registered', font_size = 38, color = [255, 255, 255, 1])
                    self.button_layout.add_widget(self.label)
            else:
                self.button_layout.remove_widget(self.label)
                self.label = Label(text='Zero/Muliple Faces Detected', font_size = 38, color = [255, 255, 255, 1])
                self.button_layout.add_widget(self.label)
        else:
            self.button_layout.remove_widget(self.label)
            self.label = Label(text='No Registered Users', font_size = 38, color = [255, 255, 255, 1])
            self.button_layout.add_widget(self.label)


    def goback(self, instance):

        if self.flag == 1:
            self.event.cancel()
            self.capture.release()
            self.remove_widget(self.layout)
            self.remove_widget(self.button_layout)
            self.__init__()
        UI_interface.screen_manager.current = "Home"


class Add_Student(GridLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_path = 'Output/'
        self.cols = 2
        self.flag = 0
        self.padding = [100, 100, 100, 100]
        self.spacing = [20, 20]
        self.begin = Button(text='BEGIN', font_size = 30, italic = True, background_color = [1, 255, 1, 1])
        self.begin.bind(on_press = self.start)
        self.add_widget(self.begin)
        self.back = Button(text='GO BACK', font_size = 30, italic = True, background_color = [255, 1, 1, 1])
        self.back.bind(on_press = self.goback)
        self.add_widget(self.back)

    def start(self, instance):
        global camera_index
        self.flag = 1
        self.img=Image()
        self.layout = BoxLayout()
        self.layout.add_widget(self.img)
        self.remove_widget(self.begin)
        self.remove_widget(self.back)
        self.add_widget(self.layout)

        self.name = TextInput(multiline = False, size_hint = (.2, None), height = 40)
        add = Button(text='ADD ME', font_size = 30, italic = True, background_color = [255, 1, 1, 1])
        back = Button(text='GO BACK', font_size = 30, italic = True, background_color = [1, 1, 255, 1])
        self.label = Label(text='Add Yourself!!', font_size = 38, color = [255, 255, 255, 1])


        add.bind(on_press = self.add)
        back.bind(on_press = self.goback)

        self.button_layout = GridLayout(rows = 4, spacing = [20, 20])

        self.button_layout.add_widget(self.name)
        self.button_layout.add_widget(add)
        self.button_layout.add_widget(back)
        self.button_layout.add_widget(self.label)

        self.add_widget(self.button_layout)

        self.capture = cv2.VideoCapture(camera_index)
        self.event = Clock.schedule_interval(self.update, 1.0/33.0)


    def update(self, instance):
        _, self.frame = self.capture.read()
        self.frame = extract_all_faces(self.frame)
        buf1 = cv2.flip(self.frame, 0)
        buf = buf1.tostring()
        texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img.texture = texture


    def add(self, instance):
        if len(self.name.text) != 0:
            ts = datetime.datetime.now()
            img_name = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
            img_path = self.output_path + img_name
            cv2.imwrite(img_path, self.frame)
            print("[INFO] saved {}".format(img_name))
            embedding, flag = generate_embedding(img_path)
            if flag == 1:
                insertBLOB(self.name.text, embedding)
                self.button_layout.remove_widget(self.label)
                self.label = Label(text=self.name.text + ' Added', font_size = 38, color = [255, 255, 255, 1])
                self.button_layout.add_widget(self.label)
            else:
                self.button_layout.remove_widget(self.label)
                self.label = Label(text = 'Zero/Multiple Faces Detected', font_size = 38, color = [255, 255, 255, 1])
                self.button_layout.add_widget(self.label)
        else:
            self.button_layout.remove_widget(self.label)
            self.label = Label(text = 'Enter UserName', font_size = 38, color = [255, 255, 255, 1])
            self.button_layout.add_widget(self.label)

    def goback(self, instance):
        global usernames, embeddings, day

        if self.flag == 1:
            self.event.cancel()
            self.capture.release()
            self.remove_widget(self.layout)
            self.remove_widget(self.button_layout)
            self.__init__()

            embeddings, usernames = readAllBlobData()
            for username in usernames:
                if username not in attendance.keys():
                    attendance[username] = 'Absent'
                    insert_data(username, 0, day)
        print(attendance)
        UI_interface.screen_manager.current = "Home"



class Show_Attendance_Page(GridLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flag = 0
        self.cols = 2
        self.padding = [100, 100, 100, 100]
        self.spacing = [20, 20]


    def show(self):
        global day

        self.attendance_list = GridLayout(cols = 2, rows = 42, spacing = [20, 20])
        for key in attendance.keys():
            self.attendance_list.add_widget(Label(text = key, font_size = 20, color = [255, 255, 255, 1]))
            self.attendance_list.add_widget(Label(text = attendance[key], font_size = 20, color = [255, 255, 255, 1]))

        self.add_widget(self.attendance_list)

        self.back = Button(text='GO BACK', font_size = 30, italic = True, background_color = [255, 1, 1, 1])
        self.back.bind(on_press = self.goback)
        #self.add_widget(self.back)
        self.button_layout = GridLayout(rows = 4, spacing = [20, 20])
        self.button_layout.add_widget(self.back)
        if day == 'day_1':
            pass

        else:
            self.visualize = Button(text='VISUALIZE', font_size = 30, italic = True, background_color = [255, 1, 1, 1])
            self.visualize.bind(on_press = self.govisual)
            self.button_layout.add_widget(self.visualize)

        self.add_widget(self.button_layout)

    def govisual(self, instance):
        self.remove_widget(self.button_layout)
        self.remove_widget(self.attendance_list)
        global show_visual_obj
        show_visual_obj.show()
        UI_interface.screen_manager.current = "Visualize"

    def goback(self, instance):

        self.remove_widget(self.button_layout)
        self.remove_widget(self.attendance_list)
        UI_interface.screen_manager.current = "Home"

class Visualize(GridLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 2
        self.padding = [100, 100, 100, 100]
        self.spacing = [20, 20]


    def show(self):
        self.im1 = Image(source = 'bar.jpg')
        self.im2 = Image(source = 'pie.jpg')
        self.add_widget(self.im1)
        self.add_widget(self.im2)
        self.back = Button(text='GO BACK', font_size = 30, size = (10, 10), italic = True, background_color = [255, 1, 1, 1])
        self.back.bind(on_press = self.goback)
        self.add_widget(self.back)

    def goback(self, instance):
        self.remove_widget(self.im1)
        self.remove_widget(self.im2)
        self.remove_widget(self.back)
        global usernames, embeddings
        global show_attendance_obj
        embeddings, usernames = readAllBlobData()
        show_attendance_obj.show()
        UI_interface.screen_manager.current = "Show_Attendance"


class Remove_Student_Page(GridLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flag = 0
        self.cols = 2
        self.padding = [100, 100, 100, 100]
        self.spacing = [20, 20]


    def show(self):
        self.name = TextInput(multiline = False, size_hint = (.2, None), height = 40)
        self.remove = Button(text='REMOVE USER', font_size = 30, italic = True, background_color = [255, 1, 1, 1])
        self.remove.bind(on_press = self.remove_student)
        self.back = Button(text='GO BACK', font_size = 30, italic = True, background_color = [255, 1, 1, 1])
        self.back.bind(on_press = self.goback)
        self.buttons = GridLayout(cols = 1, spacing = [20, 20])
        self.buttons.add_widget(self.name)
        self.buttons.add_widget(self.remove)
        self.buttons.add_widget(self.back)
        self.add_widget(self.buttons)

        self.attendance_list = GridLayout(cols = 2, rows = 42, spacing = [20, 20])
        for key in attendance.keys():
            self.attendance_list.add_widget(Label(text = key, font_size = 20, color = [255, 255, 255, 1]))
            self.attendance_list.add_widget(Label(text = attendance[key], font_size = 20, color = [255, 255, 255, 1]))

        self.add_widget(self.attendance_list)

    def remove_student(self, instance):
        if len(self.name.text) != 0:
            deleteBlob(self.name.text)
            self.remove_widget(self.attendance_list)
            if self.name.text in attendance:
                del attendance[self.name.text]
                delete_data(self.name.text)
            print(attendance)
            self.attendance_list = GridLayout(cols = 2, rows = 42, spacing = [20, 20])
            for key in attendance.keys():
                self.attendance_list.add_widget(Label(text = key, font_size = 20, color = [255, 255, 255, 1]))
                self.attendance_list.add_widget(Label(text = attendance[key], font_size = 20, color = [255, 255, 255, 1]))

            self.add_widget(self.attendance_list)


    def goback(self, instance):
        self.remove_widget(self.buttons)
        self.remove_widget(self.attendance_list)
        UI_interface.screen_manager.current = "Home"



class AIAMS(App):

    def build(self):
        global show_attendance_obj, remove_student_obj, show_visual_obj
        self.screen_manager = ScreenManager()

        self.day_select = Day_Select()
        screen = Screen(name='Day')
        screen.add_widget(self.day_select)
        self.screen_manager.add_widget(screen)

        self.home_page = Home_Page()
        screen = Screen(name='Home')
        screen.add_widget(self.home_page)
        self.screen_manager.add_widget(screen)

        self.attendance_page = Attendance_Page()
        screen = Screen(name='Attendance')
        screen.add_widget(self.attendance_page)
        self.screen_manager.add_widget(screen)

        self.add_student = Add_Student()
        screen = Screen(name='Add_Student')
        screen.add_widget(self.add_student)
        self.screen_manager.add_widget(screen)

        show_attendance_obj = Show_Attendance_Page()
        screen = Screen(name='Show_Attendance')
        screen.add_widget(show_attendance_obj)
        self.screen_manager.add_widget(screen)

        show_visual_obj = Visualize()
        screen = Screen(name='Visualize')
        screen.add_widget(show_visual_obj)
        self.screen_manager.add_widget(screen)

        remove_student_obj = Remove_Student_Page()
        screen = Screen(name='Remove_Student')
        screen.add_widget(remove_student_obj)
        self.screen_manager.add_widget(screen)

        return self.screen_manager


if __name__ == "__main__":

    if not os.path.exists('Output/'):
        os.makedirs('Output/')

    if not os.path.exists('Face_Database.db'):
        create_table()

    if not os.path.exists('student.db'):
        create_attend_table()

    UI_interface = AIAMS()
    UI_interface.run()

