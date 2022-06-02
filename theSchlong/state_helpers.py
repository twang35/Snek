
def get_food_score(head_pos, current_food):
    if current_food == 'no food':
        return 0
    food_pos = current_food.position

    # calculate distance to food
    return abs(food_pos[0] - head_pos[0]) + abs(food_pos[1] - head_pos[1])

