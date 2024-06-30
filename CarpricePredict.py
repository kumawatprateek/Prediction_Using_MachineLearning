
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.toast import toast
from kivy.core.window import Window
from sklearn import linear_model
import pickle
import pandas as pd
import numpy as np

# Set window size for mobile simulation
Window.size = (360, 640)

KV = '''
ScreenManager:
    MainScreen:
    CarpriceScreen:
    CarpriceResultScreen:
    HomepriceScreen:
    LocationScreen:
    ActivityScreen:
    ToDoScreen:
    SettingScreen:

<MainScreen>:
    name: "main"
    MDBoxLayout:
        orientation: 'vertical'
        md_bg_color: 0.1, 0.1, 0.1, .95
        padding: dp(16)
        spacing: dp(16)
        MDBoxLayout:
            size_hint_y: None
            height: self.minimum_height
            MDLabel:
                text: "Prediction System"
                font_style: "H5"
                theme_text_color: "Custom"
                text_color: 1, 1, 1, 1  # Magenta color (RGBA)
                size_hint_y: None
                height: self.texture_size[1]
            MDIconButton:
                icon: "bell-outline"
                theme_text_color: "Custom"
                text_color: 1, 1, 1, 1  # Magenta color (RGBA)
                pos_hint: {"center_y": 0.5}
        MDGridLayout:
            cols: 2
            spacing: dp(16)
            adaptive_height: True

            MDRaisedButton:
                orientation: 'vertical'
                size_hint: None, None
                size: dp(160), dp(160)
                on_release: app.card_clicked("carprice")
                MDLabel:
                    text: "Car Price Predict"
                    font_style: "H6"
                    halign: "center"

            MDRaisedButton:
                orientation: 'vertical'
                size_hint: None, None
                size: dp(160), dp(160)
                on_release: app.card_clicked("homeprice")
                MDLabel:
                    text: "Coming soon..."
                    font_style: "H6"
                    halign: "center"

            MDRaisedButton:
                orientation: 'vertical'
                size_hint: None, None
                size: dp(160), dp(160)
                on_release: app.card_clicked("location")
                MDLabel:
                    text: "Coming soon..."
                    font_style: "H6"
                    halign: "center"

            MDRaisedButton:
                orientation: 'vertical'
                size_hint: None, None
                size: dp(160), dp(160)
                on_release: app.card_clicked("activity")
                MDLabel:
                    text: "Coming soon..."
                    font_style: "H6"
                    halign: "center"

            MDRaisedButton:
                orientation: 'vertical'
                size_hint: None, None
                size: dp(160), dp(160)
                on_release: app.card_clicked("todo")
                MDLabel:
                    text: "Coming soon..."
                    font_style: "H6"
                    halign: "center"

            MDRaisedButton:
                orientation: 'vertical'
                size_hint: None, None
                size: dp(160), dp(160)
                on_release: app.card_clicked("setting")
                MDLabel:
                    text: "Setting"
                    font_style: "H6"
                    halign: "center"

<CarpriceScreen>:
    name: "carprice"
    MDFloatLayout:
        padding: dp(16)
        spacing: dp(16)
        md_bg_color: 0.1, 0.1, 0.1, .95
        
        MDBoxLayout:
            size_hint_y: None
            pos_hint: {'center_x': .5, 'center_y': .95}
            height: self.minimum_height
            MDIconButton:
                icon: "arrow-left"
                theme_text_color: "Custom"
                text_color: 1, 1, 1, 1  # Magenta color (RGBA)
                on_release:
                    app.root.current = "main"
                    app.root.transition.direction = "right"

        MDLabel:
            text: "Old Car Price"
            font_size: "40sp"
            pos_hint: {'center_x': .5, 'center_y': .85}
            halign: "center"
            theme_text_color: "Custom"
            text_color: 1, 1, 1, 1

        MDTextField:
            id: nameCar
            pos_hint: {'center_x': .5, 'center_y': .7}
            size_hint_x: None
            width: "250dp"
            hint_text: "Car Name"
            helper_text: "Search the car name"
            helper_text_mode: "on_focus"
            text_color_normal: 0.7,1,1,1
            text_color_focus: 0.7,1,1,1
            line_color_normal: 0.7,0.5,0.9,1 
            line_color_focus: 1, 1, 1, 1
            hint_text_color_normal: 0.9,1,0.2,1
            hint_text_color_focus: 1,1,0.1, 1 

        MDTextField:
            id: year
            pos_hint: {'center_x': .3, 'center_y': .6}
            size_hint_x: None
            width: "100dp"
            hint_text: "Year"
            helper_text: "Enter the year of buy car"
            helper_text_mode: "on_focus"
            max_text_length: 4
            text_color_normal: 0.7,1,1,1
            text_color_focus: 0.7,1,1,1
            line_color_normal: 0.7,0.5,0.9,1 
            line_color_focus: 1, 1, 1, 1
            hint_text_color_normal: 0.9,1,0.2,1
            hint_text_color_focus: 1,1,0.1, 1 

        MDTextField:
            id: mileage
            pos_hint: {'center_x': .7, 'center_y': .6}
            size_hint_x: None
            width: "100dp"
            hint_text: "Mileage" 
            helper_text: "Enter the year of buy car"
            helper_text_mode: "on_focus"
            max_text_length: 2
            text_color_normal: 0.7,1,1,1
            text_color_focus: 0.7,1,1,1
            line_color_normal: 0.7,0.5,0.9,1 
            line_color_focus: 1, 1, 1, 1
            hint_text_color_normal: 0.9,1,0.2,1
            hint_text_color_focus: 1,1,0.1, 1     

        MDTextField:
            id: kilometre
            pos_hint: {'center_x': .35, 'center_y': .5}
            size_hint_x: None
            width: "150dp"
            hint_text: "KiloMetre Runs"
            text_color_normal: 0.7,1,1,1
            text_color_focus: 0.7,1,1,1
            line_color_normal: 0.7,0.5,0.9,1 
            line_color_focus: 1, 1, 1, 1
            hint_text_color_normal: 0.9,1,0.2,1
            hint_text_color_focus: 1,1,0.1, 1  
            
        MDTextField:
            id: owner
            pos_hint: {'center_x': .75, 'center_y': .5}
            size_hint_x: None
            width: "70dp"
            hint_text: "Owner" 
            # helper_text: "Enter the car first,second onership"
            helper_text_mode: "on_focus"
            max_text_length: 1
            text_color_normal: 0.7,1,1,1
            text_color_focus: 0.7,1,1,1
            line_color_normal: 0.7,0.5,0.9,1 
            line_color_focus: 1, 1, 1, 1
            hint_text_color_normal: 0.9,1,0.2,1
            hint_text_color_focus: 1,1,0.1, 1        

        MDTextField:
            id: transmission
            pos_hint: {'center_x': .5, 'center_y': .4}
            size_hint_x: None
            width: "200dp"
            hint_text: "Transmission"    
            text_color_normal: 0.7,1,1,1
            text_color_focus: 0.7,1,1,1
            line_color_normal: 0.7,0.5,0.9,1 
            line_color_focus: 1, 1, 1, 1
            hint_text_color_normal: 0.9,1,0.2,1
            hint_text_color_focus: 1,1,0.1, 1     

        MDTextField:
            id: fuel
            pos_hint: {'center_x': .5, 'center_y': .3}
            size_hint_x: None
            width: "200dp"
            hint_text: "Fuel"    
            text_color_normal: 0.7,1,1,1
            text_color_focus: 0.7,1,1,1
            line_color_normal: 0.7,0.5,0.9,1 
            line_color_focus: 1, 1, 1, 1
            hint_text_color_normal: 0.9,1,0.2,1
            hint_text_color_focus: 1,1,0.1, 1      

        MDRoundFlatIconButton:
            pos_hint: {'center_x': .5, 'center_y': .1}
            text: "Find Price"
            icon: "car"
            text_color: 0.1,0,0, 1
            icon_color: 0.1,0,0.8, 1
            md_bg_color: 0.5,0.6,1, 1
            on_release: app.find_price()

<CarpriceResultScreen>:
    name: "carprice_result"
    MDBoxLayout:
        md_bg_color: 0.1, 0.1, 0.1, .95
        orientation: 'vertical'
        padding: dp(16)
        spacing: dp(16)
        
        MDIconButton:
            icon: "arrow-left"
            theme_text_color: "Custom"
            text_color: 1, 1, 1, 1  # Magenta color (RGBA)
            on_release:
                app.root.current = "carprice"
                app.root.transition.direction = "right"
        
        MDLabel:
            id: result_label
            pos_hint: {'center_x': .5, 'center_y': .6}
            text: ""
            theme_text_color: "Custom"
            text_color: 1, 1, 1, 1  # Magenta color (RGBA)
            font_style: "H4"
            halign: "center"
        
        MDIcon:
            icon: "car"
            icon_color: 0.1, 0, 0.8, 1
            size: dp(40), dp(40)
            pos_hint: {'center_x': .5, 'center_y': .3} 
            

<HomepriceScreen>:
    name: "homeprice"
    MDBoxLayout:
        orientation: 'vertical'
        padding: dp(16)
        spacing: dp(16)
        MDLabel:
            text: "Homeprice"
            font_style: "H4"
            halign: "center"
        MDRaisedButton:
            text: "Go Back"
            on_release:
                app.root.current = "main"
                app.root.transition.direction = "right"

<LocationScreen>:
    name: "location"
    MDBoxLayout:
        orientation: 'vertical'
        padding: dp(16)
        spacing: dp(16)
        MDLabel:
            text: "Location"
            font_style: "H4"
            halign: "center"
        MDRaisedButton:
            text: "Go Back"
            on_release:
                app.root.current = "main"
                app.root.transition.direction = "right"

<ActivityScreen>:
    name: "activity"
    MDBoxLayout:
        orientation: 'vertical'
        padding: dp(16)
        spacing: dp(16)
        MDLabel:
            text: "Activity"
            font_style: "H4"
            halign: "center"
        MDRaisedButton:
            text: "Go Back"
            on_release:
                app.root.current = "main"
                app.root.transition.direction = "right"

<ToDoScreen>:
    name: "todo"
    MDBoxLayout:
        orientation: 'vertical'
        padding: dp(16)
        spacing: dp(16)
        MDLabel:
            text: "To Do"
            font_style: "H4"
            halign: "center"
        MDRaisedButton:
            text: "Go Back"
            on_release:
                app.root.current = "main"
                app.root.transition.direction = "right"

<SettingScreen>:
    name: "setting"
    MDBoxLayout:
        orientation: 'vertical'
        padding: dp(16)
        spacing: dp(16)
        MDLabel:
            text: "Setting"
            font_style: "H4"
            halign: "center"
        MDRaisedButton:
            text: "Go Back"
            on_release:
                app.root.current = "main"
                app.root.transition.direction = "right"
'''


class MainScreen(Screen):
    pass


class CarpriceScreen(Screen):
    pass


class CarpriceResultScreen(Screen):
    pass


class HomepriceScreen(Screen):
    pass


class LocationScreen(Screen):
    pass


class ActivityScreen(Screen):
    pass


class ToDoScreen(Screen):
    pass


class SettingScreen(Screen):
    pass


class TestApp(MDApp):

    def build(self):
        return Builder.load_string(KV)

    def card_clicked(self, screen_name):
        self.root.current = screen_name
        self.root.transition.direction = "left"

    def car_price(self):
        car_value_name=pd.read_csv("carnames.csv")

        # Dummy implementation for car price finding
        car_name = self.root.get_screen('carprice').ids.nameCar.text
        year = self.root.get_screen('carprice').ids.year.text
        mileage = self.root.get_screen('carprice').ids.mileage.text
        kilometre = self.root.get_screen('carprice').ids.kilometre.text
        owner= self.root.get_screen('carprice').ids.owner.text
        transmission = self.root.get_screen('carprice').ids.transmission.text
        fuel = self.root.get_screen('carprice').ids.fuel.text

        id = np.array(car_value_name[car_value_name[car_name] == 1]).flatten()

        # CNG	Diesel	LPG	Petrol
        num_ful=[0,0,0,0]
        if fuel=="CNG": num_ful=[1,0,0,0]
        elif fuel=="Diesel": num_ful=[0,1,0,0]
        elif fuel == "LPG": num_ful=[0,0,1,0]
        elif fuel == "Petrol": num_ful=[0,0,0,1]

        #nums
        y=int(year)
        m=int(mileage)
        k=int(kilometre)
        o=int(owner)

        # Automatic	Manual
        num_tran=[0,0]
        if transmission=="Automatic": num_tran=[1,0]
        elif transmission=="Manual": num_tran=[0,1]

        a = np.concatenate((id, np.array([y, k, m, o]),num_ful,num_tran))

        with open('carPraicePridect_pickle', 'rb') as f:
            model_price = pickle.load(f)
        price=model_price.predict([a])
        # result = f"Price for {car_name} ({year}), Mileage: {mileage}, KM: {kilometre}, Transmission: {transmission}, Fuel: {fuel}"

        return price

    def find_price(self):

        x=self.car_price()
        result = f"Price for car {x[0]}"

        self.root.get_screen('carprice_result').ids.result_label.text = result
        self.clear_fields()
        self.root.current = "carprice_result"
        self.root.transition.direction = "left"

    def clear_fields(self):
        self.root.get_screen('carprice').ids.nameCar.text = ''
        self.root.get_screen('carprice').ids.year.text = ''
        self.root.get_screen('carprice').ids.mileage.text = ''
        self.root.get_screen('carprice').ids.kilometre.text = ''
        self.root.get_screen('carprice').ids.transmission.text = ''
        self.root.get_screen('carprice').ids.fuel.text = ''
        self.root.get_screen('carprice').ids.owner.text = ''


TestApp().run()
