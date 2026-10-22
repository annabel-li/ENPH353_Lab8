import cv2
import rclpy
import time
import gym_gazebo
import gymnasium as gym
import numpy as np
from cv_bridge import CvBridge
from gym_gazebo.core.gazebo_env import GazeboEnv
from rclpy.node import Node as RosNode
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import Twist as RosTwist

NUM_OBS_STATES = 3
NUM_ACTIONS = 3

DRIVE_LEFT = 0 
DRIVE_FWD = 1 
DRIVE_RIGHT = 2

class LineFollowerEnv(GazeboEnv):
    def __init__(self):
        # Intializing ROS2
        rclpy.init(args=None)
        
        super(LineFollowerEnv, self).__init__(launch_pkg='linefollow_ros',
                                    launch_file='my_agent.launch.xml',
                                    world_name='monza_gym')

        # Setting up ROS2 node
        self.ros_node = RosNode('line_follower_RL')

        # Initializing variables
        self.is_resetting = False
        self.latest_obs = None
        self.new_obs_event = False

        # Setting the observation and action spaces
        self.observation_space = gym.spaces.Discrete(NUM_OBS_STATES)
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        # ROS2 topics
        observation_topic = "/camera/image_raw"
        velocity_topic = "/cmd_vel"

        # Publishing and subscribing to ROS2 topics
        self.pub_cmd_vel_msg = self.ros_node.create_publisher(
            RosTwist, velocity_topic, 10)
            
        self.sub_obs = self.ros_node.create_subscription(
            RosImage, observation_topic, self.callback_obs_feed_, 10)


    def callback_obs_feed_(self, msg: RosImage):
        '''
        @brief observation callback resets the new_obs_event flag
        @param msg: Image from Gazebo sim
        '''
        # Takes in observation, returns a flag an observation is recieved
        if not self.is_resetting:
            self.latest_obs = msg
            self.new_obs_event = True


    ## MAIN FUNCTIONS:
    def reset(self, seed=None, options=None):
        '''
        @brief resets the simulation
        '''
        self.is_resetting = True
        print("Resetting")

        # Zero out velocities
        vel_cmd = RosTwist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.pub_cmd_vel_msg.publish(vel_cmd)

        # Reset position and joints
        self._reset_agents()

        # Unpause the sim
        self._pause_sim(False)

        # Allow some time to settle
        time.sleep(0.1) # Wait 100 ms
        
        self.is_resetting = False

        # Wait for next observation to come in before kickstarting the data loop
        self.new_obs_event = False
        while not self.new_obs_event:
            # Telling executor (node) to spin
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        # # Pause the sim after the first observation comes in
        self._pause_sim(True)
        
        state, _ = self.process_obs(self.latest_obs)

        print("Done resetting")
        return state, {}
    
    
    def step(self, action: int):
        '''
        @brief takes a step in the simulation
        @param action
        @retval state
        @retval reward
        @retval terminated
        @retval truncated
        '''
        # Process the action
        vel_cmd = self.process_action(action)

        # Unpause the sim
        self._pause_sim(False)

        # Moving the agent
        self.pub_cmd_vel_msg.publish(vel_cmd)

        # Leave running until next observation comes in
        self.new_obs_event = False
        while not self.new_obs_event:
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        # Pause the simulation after the next observation comes in
        self._pause_sim(True)

        state, terminated = self.process_obs(self.latest_obs)
        
        reward = self.process_reward(action, state, terminated)

        truncated = False # Used if we ran maximum number of steps.
        return state, reward, terminated, truncated, {}

    
    ## HELPER FUNCTIONS
    def process_action(self, action: int):
        '''
        @brief Translate the action the robot is taking into a Twist message
        @param action: an integer where 0 is left, 1 is forward and 2 is right. 
        If an action outside of this range is passed, the robot will stop. 
        @retval vel_cmd: Twist message equivalent of the action
        '''

        vel_cmd = RosTwist() #create a Twist object 
        angular_speed = 3.0
        drive_speed = 1.0

        if action == DRIVE_LEFT: 
            vel_cmd.angular.z = angular_speed #turn left 
        elif action == DRIVE_FWD: 
            vel_cmd.linear.x = drive_speed 
        elif action == DRIVE_RIGHT: 
            vel_cmd.angular.z = -angular_speed #turn right 
        else: #robot stops. 
            vel_cmd.angular.z = 0.0
            vel_cmd.linear.x = 0.0  

        return vel_cmd


    def process_reward(self, action, state, terminated):
        '''
        @brief depending on the action, state and terminated values assign
               a reward
        @retval reward: the reward for the current step'''

        if terminated:
                return -10
        
        if state == DRIVE_FWD and action == DRIVE_FWD:
            return 10 

        elif state == DRIVE_LEFT and action == DRIVE_LEFT or state == DRIVE_RIGHT and action == DRIVE_RIGHT:
            return 5 

        else: 

            return 0 


    """From Lab 2:"""

    def longestConnectedColour(self, imgRow, targetColour):
        """!
        @brief Finds the longest connected string of pixels in an image's row with pixel value equal to targetColour.

        @param imgRow An array representing the pixel values in the row of an image. Must be grayscale.
        @param targetColour An int representing the target colour.

        @return An array storing the index numbers of the longest connected strip of targetColour in that row.
        """
        maxLen = 0
        currentIndices = []
        longestString = []

        # if the current pixel is the target colour, add it to the list of indices
        for i in range(len(imgRow)):
            if imgRow[i] == targetColour:
                currentIndices.append(i)
            # else, we have either reached the end of the strip and should check for the strip's length
            else:
                if len(currentIndices) > maxLen:
                    maxLen = len(currentIndices)
                    longestString = currentIndices
                currentIndices = []  # reset regardless of whether it was the longest

        # save the list if it was the longest
        if len(currentIndices) > maxLen:
            longestString = currentIndices

        return longestString

    def findRoadCenter(self, processedFrame, targetColour):

        """!
        @pre processedFrame must be grayscale or black and white.

        @param processedFrame The grayscale frame of the original video.
        @param targetColour An int representing the target colour.

        @return coordinates of the bottom center of the road 
        if the road is not found the function returns (-1, -1)
        """
        # Get the image dimensions
        height, width = processedFrame.shape
            # start from bottom row, go upward
        for row in range(height - 1, -1, -1):
            # only worth checking rows that actually contain the colour
            if np.any(processedFrame[row] == targetColour):
                road = self.longestConnectedColour(processedFrame[row], targetColour)

                # get the x/y coordinates for the center of the road
                cx = road[len(road) // 2]
                cy = row

                return cx, cy 

        return (-1, -1)
        


    def process_obs(self, obs: RosImage):
        '''
        @brief Identifies the line in the image and populates the state array
               based on the line position (i.e. state=0 line is left, 
               state=1 line is in the center)
        @param obs raw image from Gazebo sim
        @retval state: the line position as a value: 
            -1 (no line), 0 (left), 1 (center), 2 (right)
        @retval terminated: the termination condition occured - the line is no 
                    longer in the field of the camera
        '''
        cv_image = CvBridge().imgmsg_to_cv2(obs, desired_encoding='bgr8')
        cv2.imshow("Camera Feed", cv_image)
        cv2.waitKey(1)

        state = 0
        terminated = False

        #get the dimensions of the image 
        im_height, im_width, _ = cv_image.shape 

        #split the image into 3 sections 
        left_section = np.arange(0, int(im_width/3))
        center_section = np.arange(int(im_width/3), int(im_width*2/3))

        #from Lab 2: 
        # Convert BGR to grayscale. 'grayFrame' is a new copy of the original frame, so the information about the original colours are preserved in 'frame'
        grayFrame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to make the road 'pop' --> use THRESH_BINARY_INV so pixels ABOVE threshold become black (0)
        _, thresh = cv2.threshold(grayFrame, 60, 255, cv2.THRESH_BINARY_INV)

        #find the road center
        roadX, roadY = self.findRoadCenter(thresh, 255)

        if roadX != -1: 
            if roadX in left_section: 
                state = 0 
            elif roadX in center_section:
                state=1 
            else:
                state=2

        #no road found. 
        else:
            state = -1 
            terminated = True 

        return state, terminated


    # Shutting down ROS2
    def user_close(self):
        self.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()