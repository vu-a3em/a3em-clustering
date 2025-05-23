import numpy as np

tests = np.array([
    [
        [12445, 1510, 1482, 1549, 1582, 1592],
        [2137, 1173, 1408, 1390, 1125, 1072],
        [1606, 1150, 1408, 1391, 1099, 1065],
        [8127, 1062, 1319, 1033, 993, 860],
        [3492, 759, 1020, 968, 786, 693],
    ],
    [
        [1537, 12401, 1565, 1542, 1618, 1497],
        [525, 645, 964, 970, 596, 542],
        [950, 492, 1038, 1070, 554, 432],
        [1208, 694, 1309, 914, 706, 559],
        [864, 373, 960, 858, 603, 528],
    ],
    [
        [1542, 1636, 12319, 1568, 1566, 1529],
        [694, 1161, 2733, 678, 971, 909],
        [1493, 1199, 2535, 569, 957, 880],
        [1267, 1172, 7112, 1033, 1000, 820],
        [1008, 864, 3849, 888, 770, 585],
    ],
    [
        [1543, 1627, 1479, 12404, 1569, 1538],
        [656, 1109, 487, 1853, 936, 799],
        [1368, 1146, 423, 1670, 889, 863],
        [1230, 986, 1282, 2885, 813, 643],
        [930, 829, 905, 1767, 736, 571],
    ],
    [
        [1514, 1551, 1547, 1626, 12322, 1600],
        [509, 519, 662, 604, 280, 443],
        [948, 358, 659, 644, 247, 353],
        [1199, 804, 1277, 914, 472, 485],
        [797, 649, 893, 818, 69, 499],
    ],
    [
        [1527, 1553, 1532, 1571, 1601, 12376],
        [531, 642, 738, 738, 624, 281],
        [495, 500, 624, 670, 463, 181],
        [1195, 776, 1253, 928, 642, 319],
        [804, 615, 827, 774, 579, 51],
    ],
], dtype = np.float64)
tests /= np.sum(tests, axis = -1, keepdims = True)

uniform = np.ones((tests.shape[-1],)) / tests.shape[-1]
for i, test in enumerate(tests):
    print(f'test {i+1}:')

    print('kl-div', ' & '.join([f'{np.sum(test[i] * np.log2(test[i] / uniform)):.4f}' for i in range(len(test))]))

    print()
