import numpy as np

def is_at_goal(target_x, target_y):
    at_goal = np.zeros_like(target_x, dtype=bool)
    mask = (target_x > 4500 ) & (target_y < 750) & (target_y > -750)
    at_goal[mask] = True

    return at_goal

def calculate_reward(absolute_obs):
    goal_x = 4800
    goal_y = 0

    robot_x = absolute_obs[:,0]
    robot_y = absolute_obs[:,1]
    target_x = absolute_obs[:,2]
    target_y = absolute_obs[:,3]
    robot_angle = absolute_obs[:,4]
    robot_angle = np.degrees(robot_angle) % 360

    angle_robot_ball = np.arctan2(target_y - robot_y, target_x - robot_x)
    angle_robot_ball = np.degrees(angle_robot_ball) % 360

    is_facing_ball = abs(angle_robot_ball - robot_angle) < 30

    robot_location = np.array([robot_x, robot_y]).T
    target_location = np.array([target_x, target_y]).T
    goal_location = np.tile(np.array([goal_x, goal_y]), (absolute_obs.shape[0], 1))

    distance_robot_target = np.linalg.norm(target_location - robot_location, axis=-1)
    distance_target_goal = np.linalg.norm(goal_location - target_location, axis=-1)

    reward_dist_to_ball = 1/distance_robot_target
    reward_dist_to_goal = 1/distance_target_goal
    reward = 0.9*reward_dist_to_goal + 0.1*reward_dist_to_ball

    at_goal = is_at_goal(target_x, target_y)

    reward[at_goal] += 1
    reward[~is_facing_ball] = 0

    return reward, at_goal


def convert_to_absolute_obs(obs):

    x_scale = 9000
    y_scale = 6000
    goal_x = 4800
    goal_y = 0

    target_x = (goal_x - obs[:, 2]*x_scale)
    target_y = (goal_y - obs[:, 3]*y_scale)
    robot_x = (target_x - obs[:, 0]*x_scale)
    robot_y = (target_y - obs[:, 1]*y_scale)

    relative_x = target_x - robot_x
    relative_y = target_y - robot_y
    relative_angle = np.arctan2(relative_y, relative_x)
    relative_angle[relative_angle < 0] += 2*np.pi

    relative_angle_minus_robot_angle = np.arctan2(obs[:, 4], obs[:, 5])
    relative_angle_minus_robot_angle[relative_angle_minus_robot_angle < 0] += 2*np.pi

    robot_angle = relative_angle - relative_angle_minus_robot_angle
    robot_angle[robot_angle < 0] += 2*np.pi

    return np.array([
        robot_x,
        robot_y,
        target_x,
        target_y,
        robot_angle
    ]).T


def convert_to_relative_obs(obs):

    x_scale = 9000
    y_scale = 6000
    goal_x = 4800
    goal_y = 0

    robot_x = obs[:, 0]
    robot_y = obs[:, 1]
    target_x = obs[:, 2]
    target_y = obs[:, 3]
    relative_x = target_x - robot_x
    relative_y = target_y - robot_y
    relative_angle = np.arctan2(relative_y, relative_x)
    relative_angle[relative_angle < 0] += 2*np.pi

    robot_angle = obs[:, 4]
    robot_angle[robot_angle < 0] += 2*np.pi

    goal_delta_x = goal_x - robot_x
    goal_delta_y = goal_y - robot_y

    goal_relative_angle = np.arctan2(goal_delta_y, goal_delta_x)
    goal_relative_angle[goal_relative_angle < 0] += 2*np.pi

    return np.vstack([
        (target_x - robot_x) / x_scale,
        (target_y - robot_y) / y_scale,
        (goal_x - target_x) / x_scale,
        (goal_y - target_y) / y_scale,
        np.sin(relative_angle - robot_angle),
        np.cos(relative_angle - robot_angle),
        np.sin(goal_relative_angle - robot_angle),
        np.cos(goal_relative_angle - robot_angle),
    ]).T

def check_valid(env, aug_obs, aug_action, aug_reward, aug_next_obs, render=False, verbose=False):

    env.reset()

    valid = True
    for i in range(len(aug_obs)):
        # env.set_abstract_state(aug_obs[i])
        env.state = env.unwrapped.state = aug_obs[i]

        next_obs, reward, done, info = env.step(aug_action[i])

        if render:
            env.render()
        # print(f'keys {info.keys()}')
        # Augmented transitions at the goal are surely not valid, but that's fine.
        if len(info)>0 and not info['is_success']:
            if not np.allclose(next_obs, aug_next_obs[i], atol=1e-5):
                valid = False
                if verbose:
                    print(f'{i}, true next obs - aug next obs', aug_next_obs[i]-next_obs)
                    print(f'{i}, true next obs', next_obs)
                    print(f'{i}, aug next obs', aug_next_obs[i])

                    # print(aug_next_obs[i, 2:4], next_obs[2:4])

            if not np.isclose(reward, aug_reward[i], atol=1e-5):
                valid = False
                if verbose:
                    print(f'{i}, aug reward: {aug_reward[i]}\ttrue reward: {reward}')

        if not valid:
            break

    return valid