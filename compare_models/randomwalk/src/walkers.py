#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Class to represent a walker with all functions to simulate walking"""

import numpy as np
import math
import pandas as pd
from datetime import timedelta
from shapely.geometry import Point
from typing import Dict, Any


class Walker:
    """Walker geometry"""

    def __init__(self, agent_id, start_time, total_steps, step_size, tessellation):
        """
        Constructs a Walker object with a randomly chosen starting point (x, y) and speed
        :param total_steps: the number of steps the walker has to walk
        :param n_walkers: the number of walkers that have to be created
        """
        self.agent_id = agent_id
        self.steps = (
            total_steps + 1
        )  # one step should be array[1], not array[0], therefore add 1 step
        # the speed is randomly chosen in the range between min_speed and max_speed for every walker created (there can be Superhumans with a value up to +2.5% over max_speed)
        self.step_size = step_size
        self.grid_bounds = {'edge': {}, 'corner': {}}
        self.grid = self.process_grid(tessellation)
        self.current_grid_id = None
        self.start_time = start_time
        self.timestamps = self.generate_timestamps()
        self.min_speed = 1
        self.max_speed = 3
        self.walker_speed = self.get_random_speed()
        # creating two arrays for containing x and y coordinates and fill them with 0s, first value of the arrays is the randomly chosen startpoint respectively
        self.x = np.zeros(self.steps)
        self.y = np.zeros(self.steps)
        self.trajectory = []
        self._check_arguments()

    def generate_timestamps(self):
        return [pd.Timestamp(self.start_time) + timedelta(hours=i) for i in range(self.steps)]

    def process_grid(self, grid):
        grid['x'], grid['y'] = grid.centroid.x, grid.centroid.y

        bounds = grid.bounds
        min_x, max_x = bounds.minx.min(), bounds.maxx.max()
        min_y, max_y = bounds.miny.min(), bounds.maxy.max()

        self.grid_bounds: Dict[str, Dict[str, Any]] = {
            'edge': {
                'left': grid[bounds.minx == min_x],
                'right': grid[bounds.maxx == max_x],
                'bottom': grid[bounds.miny == min_y],
                'top': grid[bounds.maxy == max_y],
            },
            'corner': {
                'right_top': grid[(bounds.maxx == max_x) & (bounds.maxy == max_y)],
                'left_top': grid[(bounds.minx == min_x) & (bounds.maxy == max_y)],
                'right_bottom': grid[(bounds.maxx == max_x) & (bounds.miny == min_y)],
                'left_bottom': grid[(bounds.minx == min_x) & (bounds.miny == min_y)],
            }
        }

        self.grid_bounds['edge']['left'] = self.grid_bounds['edge']['left'][bounds.maxy != max_y]
        self.grid_bounds['edge']['top'] = self.grid_bounds['edge']['top'][bounds.minx != min_x]
        self.grid_bounds['edge']['right'] = self.grid_bounds['edge']['right'][bounds.maxy != max_y]
        self.grid_bounds['edge']['top'] = self.grid_bounds['edge']['top'][bounds.maxx != max_x]
        self.grid_bounds['edge']['right'] = self.grid_bounds['edge']['right'][bounds.miny != min_y]
        self.grid_bounds['edge']['bottom'] = self.grid_bounds['edge']['bottom'][bounds.maxx != max_x]
        self.grid_bounds['edge']['left'] = self.grid_bounds['edge']['left'][bounds.miny != min_y]
        self.grid_bounds['edge']['bottom'] = self.grid_bounds['edge']['bottom'][bounds.minx != min_x]

        for edge in self.grid_bounds['edge']:
            self.grid_bounds['edge'][edge] = self.grid_bounds['edge'][edge]['tessellation_id'].values

        for corner in self.grid_bounds['corner']:
            self.grid_bounds['corner'][corner] = self.grid_bounds['corner'][corner]['tessellation_id'].values
        return grid

    def _check_arguments(self):
        """Test whether the types of values for the creation of the objects are correct"""
        if (
            not isinstance(self.steps, (int))
            and isinstance(self.startx, (int))
            and isinstance(self.starty, (int))
            and isinstance(self.walker_speed, (float))
        ):
            raise ValueError

    def get_random_speed(self):
        """
        Randomly choose a speed value in range min and max walkerspeed
        :return: a randomly chosen value between min_speed and max_speed + 2.5% (for superhumans to occur)
        """
        return round(
            np.random.uniform(
                self.min_speed, self.max_speed + (self.max_speed * 0.025)
            ),
            2,
        )

    def get_random_startpoint(self):
        """Randomly choose a starting point in range -sqrt and sqrt of the steps. Used to keep the plot relatively compact (responsive)."""
        grid_id = np.random.randint(0, (len(self.grid) + 1))
        x = self.grid.loc[self.grid['tessellation_id'] == grid_id, 'geometry'].iloc[0].centroid.x
        y = self.grid.loc[self.grid['tessellation_id'] == grid_id, 'geometry'].iloc[0].centroid.y
        return x, y, grid_id
        # return np.random.randint(-math.sqrt(self.steps), math.sqrt(self.steps))

    def get_random_direction(self, movepattern, posible_moveset):
        """
        Choose randomly between two movement patterns (Moore with 8 directions and Neumann with 4 directions)
        :return: a value from the specified movement set which can be used in plan_next_step()
        """
        if movepattern == "true":
            if posible_moveset:
                moveset = posible_moveset
            else:
                moveset = [
                    "East",
                    "Southeast",
                    "South",
                    "Southwest",
                    "West",
                    "Northwest",
                    "North",
                    "Northeast",
                ]
        else:
            moveset = ["East", "South", "West", "North"]
        return np.random.choice(moveset)

    def plan_next_step(self, direction, step):
        """
        Plans the next step in the Moore neighborhood by using the randomized get_random_direction()
        :param direction: return value of get_random_direction()
        :param step: index value of random_walk()
        """
        if direction == "East":
            x_temp = self.x[step - 1] + self.step_size
            y_temp = self.y[step - 1]
        elif direction == "Southeast":
            x_temp = self.x[step - 1] + self.step_size
            y_temp = self.y[step - 1] - self.step_size
        elif direction == "South":
            x_temp = self.x[step - 1]
            y_temp = self.y[step - 1] - self.step_size
        elif direction == "Southwest":
            x_temp = self.x[step - 1] - self.step_size
            y_temp = self.y[step - 1] - self.step_size
        elif direction == "West":
            x_temp = self.x[step - 1] - self.step_size
            y_temp = self.y[step - 1]
        elif direction == "Northwest":
            x_temp = self.x[step - 1] - self.step_size
            y_temp = self.y[step - 1] + self.step_size
        elif direction == "North":
            x_temp = self.x[step - 1]
            y_temp = self.y[step - 1] + self.step_size
        elif direction == "Northeast":
            x_temp = self.x[step - 1] + self.step_size
            y_temp = self.y[step - 1] + self.step_size

        grid = self.get_grid_id(x_temp, y_temp)
        self.x[step] = float(grid.geometry.iloc[0].centroid.x)
        self.y[step] = float(grid.geometry.iloc[0].centroid.y)
        self.current_grid_id = grid['tessellation_id'].values[0]

        self.trajectory.append((self.agent_id, self.timestamps[step], self.x[step], self.y[step], self.current_grid_id))

    def get_grid_id(self, x, y):
        point = Point(x, y)
        matching_cell = self.grid[self.grid.geometry.contains(point)]
        if not matching_cell.empty:
            return matching_cell
        else:
            pass

    def compute_possible_moveset(self, grid_id):
        if grid_id in self.grid_bounds['edge']['left']:
            return ['East', 'Northeast', 'North', 'Southeast', 'South']
        elif grid_id in self.grid_bounds['edge']['top']:
            return ['East', 'South', 'Southeast', 'Southwest', 'West']
        elif grid_id in self.grid_bounds['edge']['right']:
            return ['West', 'Northwest', 'North', 'Southwest', 'South']
        elif grid_id in self.grid_bounds['edge']['bottom']:
            return ['East', 'Northeast', 'North', 'Northwest', 'West']
        elif grid_id in self.grid_bounds['corner']['right_top']:
            return ['West', 'Southwest', 'South']
        elif grid_id in self.grid_bounds['corner']['left_top']:
            return ['East', 'Southeast', 'South']
        elif grid_id in self.grid_bounds['corner']['right_bottom']:
            return ['West', 'Northwest', 'North']
        elif grid_id in self.grid_bounds['corner']['left_bottom']:
            return ['East', 'Northeast', 'North']
        else:
            return None

    def random_walk(self, move_pattern, random_start):
        """Put together get_random_direction() and plan_next_step() to simulate walking"""
        if random_start == "true":
            self.x[0], self.y[0], self.current_grid_id = self.get_random_startpoint()
            # self.y[0] = self.get_random_startpoint()
            self.trajectory.append((self.agent_id, self.timestamps[0], self.x[0], self.y[0], self.current_grid_id))
        for step in range(1, self.steps):  # start with step after startpoint
            possible_moveset = self.compute_possible_moveset(self.current_grid_id)
            self.plan_next_step(self.get_random_direction(move_pattern, posible_moveset=possible_moveset), step)
            # print(self.x[step], self.y[step])

    def walker_type(self):
        """Returns the type of walker based on the randomly chosen speeds of the walker objects in the range of min and max speed"""
        calc = (
            self.max_speed - self.min_speed
        ) * 0.25  # calculate 25% of the total speed range
        if self.walker_speed < (self.min_speed + calc):  # 25% probability
            walker_type = "Slow Walker"
        elif (
            (self.min_speed + calc) <= self.walker_speed < (self.max_speed - calc)
        ):  # 50% probability
            walker_type = "Normal Walker"
        elif (
            (self.max_speed - calc) <= self.walker_speed <= self.max_speed
        ):  # 25% probability
            walker_type = "Fast Walker"
        else:
            walker_type = "Superhuman"  # can get values outside the speed range (up to +2.5% of max speed possible)
        return walker_type

    def __str__(self):
        """Returns a formatted string with details about the Walker object"""
        return "{}, Start coordinates: x={:0.2f}/y={:0.2f}, speed={}".format(
            self.walker_type(), self.x[0], self.y[0], self.walker_speed
        )
