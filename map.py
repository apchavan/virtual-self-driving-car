# Building environment for self driving car


# Import standard libraries of Python
import random
import time

# Import NumPy (used to draw the car's road with "sand")
import numpy as np

# Import Kivy packages
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Import the DQN class from "car_DQN_model.py"
from car_DQN_model import DQN


# Set configuration object API: https://kivy.org/doc/stable/api-kivy.config.html#kivy.config.ConfigParser.set

# To avoid drawing of "RED" colored dot on map when we right-click.
Config.set("input", "mouse", "mouse,multitouch_on_demand")

# To avoid resizing of map window.
Config.set("graphics", "resizable", False)


# Last X-axis draw location of sand
last_x = 0

# Last Y-axis draw location of sand
last_y = 0

# Total points in last draw
n_points = 0

# Total length of last draw
length = 0

# Last distance from car to goal
last_distance = 0

'''
	AI brain of car with '4' inputs (states), '3' outputs(actions) & 
	'0.9' gamma (a constant used in "Temporal Difference" formula to know how well 
	our car model is doing) (See line no. 107 in file "car_DQN_model.py")
'''
brain = DQN(input_size=4, nb_action=3, gamma=0.9)

'''
	Used to map selected action (either of 0, 1 or 2) to rotation in degrees. 
	Action 0 => 0° rotation (straight), 
	Action 1 => 20° rotation (left) & 
	Action 2 => -20° rotation (right)
'''
action2rotation = [0, 20, -20]

'''
	To store the reward received by performing action in current state to enter into next state
'''
reward = 0

# Boolean used to initialize car map only once
first_update = True


def init_map():
	'''
		An array of size equal to total pixels on GUI of car map; 
		also array cell value is '1' if that pixel has sand or value '0' otherwise
	'''
	global sand

	# X co-ordinate of goal
	global goal_x

	# Y co-ordinate of goal
	global goal_y

	# Map initializer boolean (declared on line no. 70 above)
	global first_update
	
	# 'sand' array initialized to all zeros with dimensions (map_width, map_height)
	sand = np.zeros((map_width, map_height))

	'''
		Initial goal (X co-ordinate) at upper left corner with X position '20' 
		(CAN NOT use position '0' because by touching the wall, car will get a bad reward; 
		wall border is at pixel 0 (left edge of window))
	'''	
	goal_x = 20

	'''
		Similarly, initial goal (Y co-ordinate) at upper left corner with Y position
	'''
	goal_y = map_height - 20

	# Indicate that the map is now initialized
	first_update = False


'''	Kivy properties classes => https://kivy.org/doc/master/api-kivy.properties.html

	(1) NumericProperty => 
		'NumericProperty' is used to bind & specify the corresponding object accepts either 'int' or 'float' type value. 
	Values of other types will throw "ValueError".
		Reference => https://kivy.org/doc/master/api-kivy.properties.html#kivy.properties.NumericProperty

	(2) ReferenceListProperty => 
		'ReferenceListProperty' is used for two NumericProperty values.
		Below on line no. 142, 145; two NumericProperty 'velocity_x' & 'velocity_y' are parameters to 
	'ReferenceListProperty' of type ReferenceListProperty. So, when we read 'velocity' it'll return 
	a tuple of 'velocity_x' & 'velocity_y'. Also if we change value of 'velocity', it'll two change 
	two values 'velocity_x' & 'velocity_y'.
		Reference => https://kivy.org/doc/master/api-kivy.properties.html#kivy.properties.ReferenceListProperty

	{#} Vector class => 
		'Vector' represent a 2D vector which provides several functions/operations for computation.
		Some of the used functions include:
			'angle()' => Compute angle between two input parameters (usually passed as tuple) & return angle in degrees.
			'distance()' => To return distance between two points (usually passed as tuple).
			'rotate()' => To rotate a vector with an angle given as parameter in degrees.
		Numeric operators of vectors include "+", "-", "/".
		Reference => https://kivy.org/doc/master/api-kivy.vector.html
'''
class Car(Widget):
	""" Create class Car """

	# Angle between X-axis of the map & axis of the car
	angle = NumericProperty(0)

	# Last rotation of the car (after playing the action, the car does a rotation of 0, 20 or -20 degrees)
	rotation = NumericProperty(0)

	# X co-ordinate of velocity vector
	velocity_x = NumericProperty(0)

	# Y co-ordinate of velocity vector
	velocity_y = NumericProperty(0)

	# Velocity vector (used to get tuple like (velocity_x, velocity_y) of previous two velocities)
	velocity = ReferenceListProperty(velocity_x, velocity_y)

	# X co-ordinate of first sensor
	sensor1_x = NumericProperty(0)

	# Y co-ordinate of first sensor
	sensor1_y = NumericProperty(0)

	# First sensor vector which will contain tuple like (sensor1_x, sensor1_y) of previous two sensors
	sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)

	# X co-ordinate of second sensor
	sensor2_x = NumericProperty(0)

	# Y co-ordinate of second sensor
	sensor2_y = NumericProperty(0)

	# Second sensor vector which will contain tuple like (sensor2_x, sensor2_y) of previous two sensors
	sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)

	# X co-ordinate of third sensor
	sensor3_x = NumericProperty(0)

	# Y co-ordinate of third sensor
	sensor3_y = NumericProperty(0)

	# Third sensor vector which will contain tuple like (sensor3_x, sensor3_y) of previous two sensors
	sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)

	# Signal received by sensor 1
	signal1 = NumericProperty(0)

	# Signal received by sensor 2
	signal2 = NumericProperty(0)

	# Signal received by sensor 3
	signal3 = NumericProperty(0)


	def move(self, rotation):
		'''
			Update position of car according to last position & velocity. 
			'*' is unzipping operator used to get tuple like (velocity_x, velocity_y) from 'self.velocity'
		'''
		self.pos = Vector(*self.velocity) + self.pos

		# Get rotation of car
		self.rotation = rotation

		# Update the angle between X-axis of the map & axis of the car
		self.angle = self.angle + self.rotation

		# Update the position of sensor 1
		self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos

		# Update the position of sensor 2
		self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos

		# Update the position of sensor 3
		self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos

		'''
			Calculating signal received by sensor 1 :=
				(1) The signal is simply the density of sand covered in the area of sensor 
				(i.e. sensor 1 here).
				(2) We have value '1' if there is sand & value '0' otherwise, in 
				each respective cell of 'sand' array.
				(3) We calculate sum of total '1' values present in sand matrix using 
				values of 'sensor1_x' & 'sensor1_y'.
				(4) Consider all sensors can sense upto 20 by 20 cells (i.e. 20x20 = pixels) around.
				(5) So, we used 'int(self.sensor1_x) - 10' from PREVIOUS 10 pixels & 
				'int(self.sensor1_x) + 10' upto NEXT 10 pixels resulting 20 pixels of X-axis.
				Similarly, 'int(self.sensor1_y) - 10' from PREVIOUS 10 pixels & 
				'int(self.sensor1_x) + 10' upto NEXT 10 pixels resulting 20 pixels of Y-axis.
				(6) Finally, to get density of total '1' values, we divide that sum by 400 
				(since sensor can sense 400 pixels i.e. 20x20)
			For sensors 2 & 3, process is same except use of their X and Y co-ordinates 
			of respective sensors.
		'''

		# Signal received by sensor 1
		self.signal1 = int(np.sum(sand[int(self.sensor1_x) - 10 : int(self.sensor1_x) + 10, int(self.sensor1_y) - 10 : int(self.sensor1_y) + 10])) / 400.0

		# Signal received by sensor 2
		self.signal2 = int(np.sum(sand[int(self.sensor2_x) - 10 : int(self.sensor2_x) + 10, int(self.sensor2_y) - 10 : int(self.sensor2_y) + 10])) / 400.0

		# Signal received by sensor 3
		self.signal3 = int(np.sum(sand[int(self.sensor3_x) - 10 : int(self.sensor3_x) + 10, int(self.sensor3_y) - 10 : int(self.sensor3_y) + 10])) / 400.0

		'''
			Check whether car or any sensor is facing EDGE of map
			'sensor1_x', 'sensor2_x', 'sensor3_x' used to check along X-axis.
			'sensor1_y', 'sensor2_y', 'sensor3_y' used to check along Y-axis.

			'self.sensorN_x > (map_width - 10)' -> To check edge or crossing MAX width of map horizontally
			'self.sensorN_x < 10' -> To check edge or crossing MIN width of map horizontally
			'self.sensorN_y > (map_height - 10)' -> To check edge or crossing MAX height of map vertically
			'self.sensorN_y < 10' -> To check edge or crossing MIN width of map vertically
			Where N above in 'sensorN_x' or 'sensorN_y' is respective sensor number along an axis.
		'''
		if self.sensor1_x > (map_width - 10) or self.sensor1_x < 10 \
		or self.sensor1_y > (map_height - 10) or self.sensor1_y < 10:
			self.signal1 = 1.0		# Signal1 get full sand density that surrounded sensor1

		if self.sensor2_x > (map_width - 10) or self.sensor2_x < 10 \
		or self.sensor2_y > (map_height -10) or self.sensor2_y < 10:
			self.signal2 = 1.0		# Signal2 get full sand density that surrounded sensor2

		if self.sensor3_x > (map_width - 10) or self.sensor3_x < 10 \
		or self.sensor3_y >(map_height - 10) or self.sensor3_y < 10:
			self.signal3 = 1.0		# Signal2 get full sand density that surrounded sensor3


'''
	We using three sensors for our car i.e. those sensors we're going to display on map 
	along with car.
	So, we MUST create classes for each of sensors by inheriting 'Widget' class in Kivy library 
	in order to bind & render them on map.
'''
class Ball1(Widget):
	""" Class for sensor 1 """
	pass

class Ball2(Widget):
	""" Class for sensor 2 """
	pass

class Ball3(Widget):
	""" Class for sensor 3 """
	pass



'''
	ObjectProperty =>
		'ObjectProperty' is used to bind objects defined in Kivy language file in a Python class.
	Reference => https://kivy.org/doc/master/api-kivy.properties.html#kivy.properties.ObjectProperty
'''
class Game(Widget):

	# Get 'car' object from "car.kv" file (see line no. 52 in "car.kv" file)
	car = ObjectProperty(None)

	# Get sensor 1 object from "car.kv" file (see line no. 53 in "car.kv" file)
	ball1 = ObjectProperty(None)

	# Get sensor 2 object from "car.kv" file (see line no. 54 in "car.kv" file)
	ball2 = ObjectProperty(None)

	# Get sensor 3 object from "car.kv" file (see line no. 55 in "car.kv" file)
	ball3 = ObjectProperty(None)


	def serve_car(self):	# Method to decide initial config (location & velocity) of car
		'''
			In the beginning put 'car' object's center (self.car.center) to 
			center of current widget (self.center)
		'''
		self.car.center = self.center

		'''
			Provide initial velocity to the car. So here, 
			car will move towards right (exact horizontally) with speed of 6.
		'''
		self.car.velocity = Vector(6, 0)


	def update(self, dt):	# The parameter 'dt' stands for "delta-time" which is used later by Kivy's 'Clock' object later to call this function.
		# AI brain of car (see line no. 54)
		global brain

		# Store reward received (see line no. 67)
		global reward

		# Last distance from car to goal (see line no. 47)
		global last_distance

		# X co-ordinate of goal
		global goal_x

		# Y co-ordinate of goal
		global goal_y

		# Width of map
		global map_width

		# Height of map
		global map_height

		map_width = self.width		# Store width of our map widget (horizontal)
		map_height = self.height	# Store height of our map widget (vertical)

		if first_update:			# If this is the first run/update, we initialize map
			init_map()

		xx = goal_x - self.car.x	# Difference between X co-ordinates between goal & car
		yy = goal_y - self.car.y	# Difference between Y co-ordinates between goal & car

		'''
			'orientation' is direction of car with respect to goal (this means, 
			if car heading perfectly towards goal, then 'orientation = 0')
		'''
		orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.0

		'''
			Now define the '4' input states for our model.
			First is orientation, which will tell whether car is heading in right direction or not.
			Second, third & fourth are respective signals received from their sensors which is 
			density of sand, calculated in 'Car' class.
		'''
		state = [orientation, self.car.signal1, self.car.signal2, self.car.signal3]

		'''
			To get an action among '3' actions, we call 'update()' method of brain, that is 
			method of DQN class which will update the weights & return the next action to perform.
			('update()' method of object 'brain' is implemented in file "car_DQN_model.py" on line no. 234)
		'''
		action = brain.update(new_state=state, new_reward=reward)

		'''
			Get the the action played (0, 1 or 2) into the rotation angle (0°, 20° or -20°)
			('action2rotation' list is defined on line no. 62)
		'''
		rotation = action2rotation[action]

		 # Actually move the car in map using 'move()' method (defined on line no. 187)
		self.car.move(rotation=rotation)

		'''
			Get the new distance between goal & car after car has been moved to new location.
				We squared the difference between (self.car.x - goal_x) & (self.car.y - goal_y) since 
			if 'goal_x' is greater than 'self.car.x' or 'goal_y' is greater than 'self.car.y', 
			the difference will be NEGATIVE & distance is NEVER NEGATIVE!
		'''
		distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)

		'''
			Update positions of three sensors decribed by objects 'ball1', 'ball2' & 'ball3' 
			using respective sensor vectors from 'car' object.
		'''
		self.ball1.pos = self.car.sensor1
		self.ball2.pos = self.car.sensor2
		self.ball3.pos = self.car.sensor3

		# If car is on sand (value will be '1' on current sand[x, y])
		if sand[int(self.car.x), int(self.car.y)] > 0:
			# Slow down the speed to 1
			self.car.velocity = Vector(1, 0).rotate(self.car.angle)

			# Also give bad (negative) reward for getting on sand/obstacle
			reward = -1
		
		else:
			# Otherwise go with normal speed
			self.car.velocity = Vector(6, 0).rotate(self.car.angle)

			# Assign slightly bad (negative) reward
			reward = -0.2

			# But if current distance to goal is smaller
			if distance < last_distance:
				# Assign a small good (positive) reward
				reward = 0.1


		if self.car.x < 10:					# If car is at left edge of map window
			self.car.x = 10					# Bring car back 10 pixels away from left edge
			reward = -1						# Give bad (negative) reward

		if self.car.x > (self.width - 10):	# If car is at right edge of map window
			self.car.x = self.width - 10	# Bring car back 10 pixels away from right edge
			reward = -1						# Give bad (negative) reward

		if self.car.y < 10:					# If car is at top edge of map window
			self.car.y = 10					# Bring car back 10 pixels away from top edge
			reward = -1						# Give bad (negative) reward

		if self.car.y > (self.height - 10):	# If car is at bottom edge of map window
			self.car.y = self.height - 10	# Bring car back 10 pixels away from bottom edge
			reward = -1						# Give bad (negative) reward

		'''
			Check if car reaches the goal.
			If True, then the point from where the car has came will be the new goal & 
			current goal will be new point of start.
			This helps to car to make round-trips in between two destinations.
		'''
		if distance < 100:
			print(f"\n\n CURRENT goal_x => {goal_x}, goal_y => {goal_y}")
			goal_x = self.width - goal_x
			goal_y = self.height - goal_y
			print(f" NEW goal_x => {goal_x}, goal_y => {goal_y}")

		last_distance = distance	# Finally store current distance as 'last_distance'


'''
	(1) on_touch_down(touch) =>
		Event called when a touch down event is initiated.
		Reference => https://kivy.org/doc/stable/api-kivy.core.window.html#kivy.core.window.WindowBase.on_touch_down

	(2) on_touch_move(touch) =>
		Event called when a touch event moves (changes location).
		Reference => https://kivy.org/doc/stable/api-kivy.core.window.html#kivy.core.window.WindowBase.on_touch_move
'''
class PaintWidget(Widget):
	""" Class used to paint on map i.e. to draw sand one map using click or touch events """
	
	def on_touch_down(self, touch):	# Event method to put some sand on map when we do left click
		global length
		global n_points
		global last_x
		global last_y

		with self.canvas:	# Use 'canvas' object to draw things in block below
			# Set drawing color
			Color(0.8, 0.7, 0)

			'''
				'touch.ud' is Python dictionary (type "<dict>") that allows us to store 
				custom attributes for a 'touch'.
			'''
			touch.ud["line"] = Line(points=(touch.x, touch.y), width=10)

			sand[int(touch.x), int(touch.y)] = 1	# Set current pixel now contains sand using value '1'
			last_x = int(touch.x)					# Store last X-axis draw location of sand
			last_y = int(touch.y)					# Store last Y-axis draw location of sand
			n_points = 0							# Set total points in last draw to '0' initially
			length = 0								# Set total length in last draw to '0' initially


	def on_touch_move(self, touch):	# Event method to draw sand when dragging with left click
		global length
		global n_points
		global last_x
		global last_y

		if touch.button == "left":	# Check if dragging with "LEFT" click only.
			# Insert new points along with existing drawn ones
			touch.ud["line"].points += [touch.x, touch.y]
			x = int(touch.x)
			y = int(touch.y)

			'''
				Calculate length of drawing (similarly as we did for 'distance' in line no. 382) 
				with atleast store length '2' using 'max()' function.
			'''
			length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
			
			# Increment total points in last draw by '1'.
			n_points += 1

			# Calculate density to use in width of line to be drawn.
			density = n_points / length

			# Set width of line beign drawn using previous density formula.
			touch.ud["line"].width = int(20 * density + 1)

			# Set all pixels (20x20) in touch region to '1' i.e. sand region.
			sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
			
			# Store last X-axis draw location of sand
			last_x = x

			# Store last Y-axis draw location of sand
			last_y = y


'''
	App => https://kivy.org/doc/master/api-kivy.app.html
		The 'App' class is base for creating Kivy applications. It's the entry point into the 
	Kivy run loop.
'''
class CarApp(App):
	""" Class to actually build & run the applicaion """

	def build(self):	# For building the app
		parent = Game()
		parent.serve_car()	# Set car's initial config (line no. 301)

		'''
			Use 'Clock' object to schedule an event to be called every <timeout> seconds, using 
			'schedule_interval()' method.
			Reference => https://kivy.org/doc/stable/api-kivy.clock.html#kivy.clock.CyClockBase.schedule_interval
		'''
		Clock.schedule_interval(callback=parent.update, timeout=(1.0 / 60.0))

		clearBtn = Button(text="Clear")
		clearBtn.bind(on_release=self.clear_canvas)

		saveBtn = Button(text="Save", pos=(parent.width, 0))
		saveBtn.bind(on_release=self.save_brain)

		loadBtn = Button(text="Load", pos=((parent.width * 2), 0))
		loadBtn.bind(on_release=self.load_brain)

		self.painter = PaintWidget()
		parent.add_widget(self.painter)
		parent.add_widget(clearBtn)
		parent.add_widget(saveBtn)
		parent.add_widget(loadBtn)
		return parent

	def clear_canvas(self, obj):
		""" Clear all the sand drawn previously """
		global sand
		self.painter.canvas.clear()		# Built-in method to clear 'canvas' object
		sand = np.zeros((map_width, map_height))

	def save_brain(self, obj):
		print("\n+1 => Saving model weights... ")
		brain.save()

	def load_brain(self, obj):
		print("\n+1 => Searching last checkpoint... ")
		brain.load()


# Run the application
if __name__ == '__main__':
	CarApp().run()
