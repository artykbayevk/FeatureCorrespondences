#%%
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.spatial.distance import euclidean
from itertools import product
import cv2
origin_p_x = 0.
origin_p_y = 0.

center_q_x = 0.
center_q_y = 0.

ell = np.linspace(0, 2 * pi, 25)



#%%
def figure_1():
    radius_on_x = 1.0
    radius_on_y = 2.5

    p_y = np.array(sorted([(origin_p_y+(0.5+0.5*y)) for y in range(0, 3)] + [(origin_p_y - (0.5 + 0.5*y)) for y in range(0, 3)] + [origin_p_y]))
    p_x = np.array([origin_p_x for x in range(0, 7)])

    plt.scatter(center_q_x + radius_on_x*np.cos(ell), center_q_y + radius_on_y*np.sin(ell), label = 'Q')
    plt.scatter(p_x, p_y, label = 'P')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis("equal")
    plt.grid(color='lightgray', linestyle='--')
    plt.legend('QP', loc='upper left')
    plt.show()

    return p_x, p_y, center_q_x + radius_on_x * np.cos(ell), center_q_y + radius_on_y * np.sin(ell)


def figure_2():
    radius_on_x = 2.5
    radius_on_y = 1.0

    p_x = np.array(sorted(
        [(origin_p_x + (0.5 + 0.5 * x)) for x in range(0, 3)] + [(origin_p_x - (0.5 + 0.5 * x)) for x in
                                                                 range(0, 3)] + [origin_p_x]))
    p_y = np.array([origin_p_y for y in range(0, 7)])

    plt.scatter(center_q_x + radius_on_x * np.cos(ell), center_q_y + radius_on_y * np.sin(ell))
    plt.scatter(p_x, p_y)
    plt.grid(color='lightgray', linestyle='--')
    plt.axis("equal")
    plt.show()
    return p_x, p_y, center_q_x + radius_on_x * np.cos(ell), center_q_y + radius_on_y * np.sin(ell)


def figure(angle, debug=False):
    radius_on_x = 1.0
    radius_on_y = 2.5

    p_y = np.array(sorted([(origin_p_y+(0.5+0.5*y)) for y in range(0, 3)] + [(origin_p_y - (0.5 + 0.5*y)) for y in range(0, 3)] + [origin_p_y]))
    p_x = np.array([origin_p_x for x in range(0, 7)])

    new_p_y = -p_x * np.sin(np.pi - angle) + p_y * np.cos(np.pi - angle)
    new_p_x = p_x * np.cos(np.pi - angle) + p_y * np.sin(np.pi - angle)

    xpos = center_q_x + radius_on_x*np.cos(ell)
    ypos = center_q_y + radius_on_y*np.sin(ell)

    new_xpos = xpos * np.cos(np.pi - angle) + ypos * np.sin(np.pi - angle)
    new_ypos = -xpos * np.sin(np.pi - angle) + ypos * np.cos(np.pi - angle)

    if debug:
        plt.scatter(new_xpos, new_ypos)
        plt.scatter(new_p_x, new_p_y)
        # plt.scatter(p_x, p_y)
        plt.axis("equal")
        plt.grid(color='lightgray', linestyle='--')
        plt.show()

    # return p_x, p_y, center_q_x + radius_on_x * np.cos(ell), center_q_y + radius_on_y * np.sin(ell)
    return new_p_x, new_p_y, new_xpos, new_ypos


def get_distances(p_x, p_y, q_x, q_y):
    p_ = [np.array([x, y]) for x, y in zip(p_x, p_y)]
    q_ = [np.array([x, y]) for x, y in zip(q_x, q_y)]
    min_dict = {}
    for p, q in [(p, q) for p in p_ for q in q_]:
        dist = float('{:.4f}'.format(euclidean(p, q)))
        if dist <= 1.0:
            key_p = '{:.2f}:{:.2f}'.format(p[0], p[1])
            key_q = '{:.2f}:{:.2f}'.format(q[0], q[1])
            if key_p not in min_dict:
                min_dict[key_p] = dist
            else:
                if dist<min_dict[key_p]:
                    min_dict[key_p] = dist
    pair_dict = {}
    for p, q in [(p, q) for p in p_ for q in q_]:
        dist = float('{:.4f}'.format(euclidean(p, q)))
        if dist <= 1.0:
            key_p = '{:.2f}:{:.2f}'.format(p[0], p[1])
            key_q = '{:.2f}:{:.2f}'.format(q[0], q[1])

            if min_dict[key_p] == dist:
                if key_p not in pair_dict:
                    pair_dict[key_p] = [key_q]
                else:
                    pair_dict[key_p].append(key_q)
    return pair_dict


def get_solutions(pairs):
    solutions = [dict(zip(pairs, v)) for v in product(*pairs.values())]
    return solutions


def draw_solution(solutions):
    for idx, item in enumerate(solutions):
    # idx = np.random.randint(0, len(solutions)-1)
    # item = solutions[idx]

        p = item.keys()
        q = item.values()
        i = 0
        colors = get_cmap(len(item))



        pairX1 = []
        pairY1 = []

        pairX2 = []
        pairY2 = []
        for p_, q_ in zip(p, q):
            x = np.array([float(p_.split(':')[0]), float(q_.split(':')[0])])
            y = np.array([float(p_.split(':')[1]), float(q_.split(':')[1])])
            plt.scatter(x, y, cmap=colors)
            plt.plot(x, y)
            plt.text(x[0]+0.03, y[0]+0.03, 'p'+str(i), fontsize=9)
            plt.text(x[1] + 0.03, y[1] + 0.03, 'q'+str(i), fontsize=9)
            i += 1

        #     if i == 1 or i == 2:
        #         pairX1.append(x[0])
        #         pairY1.append(y[0])
        #
        #         pairX2.append(x[1])
        #         pairY2.append(y[1])
        #
        # plt.plot(pairX1, pairY1, ':', c='b')
        # plt.plot(pairX2, pairY2, ':', c='y')
        plt.axis("equal")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(color='lightgray', linestyle='--')
        plt.show()


    return item


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def get_value_for_solution(sol):
    p = np.array([np.array([float(i) for i in x.split(':')]) for x in sol.keys()])[::-1]
    q = np.array([np.array([float(i) for i in x.split(':')]) for x in sol.values()])[::-1]
    output_value = 0
    iter_value = 1/(len(sol) - 1)
    for idx, item in enumerate(p):
        if idx == p.shape[0]-1:
            break
        cur_p = item
        next_p = p[idx+1]

        cur_q = q[idx]
        next_q = q[idx+1]

        dist_p = euclidean(cur_p, next_p)
        dist_q = euclidean(cur_q, next_q)

        if abs(dist_q-dist_p) < 0.3:
            output_value += iter_value+np.random.uniform(0.05, 0.10, 1000)[np.random.randint(0, 999)]

    main_out_limit = np.random.uniform(0.75, 0.85, 10000)[np.random.randint(0, 9999)]
    if output_value > main_out_limit:
        value = 1.0
    else:
        value = 0.0
    # print(output_value, main_out_limit)
    return p, q, value


def create_samples(solutions):
    data = []
    for sol in solutions:
        p, q, val = get_value_for_solution(sol)
        sample = np.array(([np.concatenate((p_, q_)) for p_, q_ in zip(p, q)])).reshape(-1)
        sample = np.append(sample, val)
        data.append(sample)
    data = np.array(data)
    return data


def main():
    p_x, p_y, q_x, q_y = figure(0)
    p_q_distances = get_distances(p_x, p_y, q_x, q_y)
    solutions = get_solutions(p_q_distances)
    samples = create_samples(solutions)
    origin_angle = np.pi / 1000
    val = np.pi / 1000
    for i in range(0, 1000):
        px_, py_, qx_, qy_ = figure(origin_angle)
        distances = get_distances(px_, py_, qx_, qy_)
        sols = get_solutions(distances)
        item_sample = create_samples(sols)

        samples = np.append(samples, item_sample, axis=0)

        origin_angle += val
    np.save('data/dataset.npy', samples)


def get_dataset():
    p_x, p_y, q_x, q_y = figure(0)
    p_q_distances = get_distances(p_x, p_y, q_x, q_y)
    solutions = get_solutions(p_q_distances)
    samples = create_samples(solutions)
    origin_angle = np.pi / 100
    val = np.pi / 100
    for i in range(0, 100):
        px_, py_, qx_, qy_ = figure(origin_angle)
        distances = get_distances(px_, py_, qx_, qy_)
        sols = get_solutions(distances)
        item_sample = create_samples(sols)

        samples = np.append(samples, item_sample, axis=0)

        origin_angle += val

    return samples

# main()

get_dataset()
def reading_dataset():
    dataset = np.load('data/dataset.npy')
    print(dataset[100000])


def temp():
    p_x, p_y, q_x, q_y = figure_1()
    p_q_distances = get_distances(p_x, p_y, q_x, q_y)
    solutions = get_solutions(p_q_distances)
    draw_solution(solutions)
    samples = create_samples(solutions)

# temp()
# reading_dataset()