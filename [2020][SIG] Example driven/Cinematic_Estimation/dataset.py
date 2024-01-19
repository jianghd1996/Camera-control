import numpy as np
import random
import torch.utils.data as data

class estimationdataset(data.Dataset):
    def __init__(self, data, label, add_noise = True):
        data0 = []
        data1 = []
        data2 = []
        data3 = []
        for v in data:
            data0 += v[0]
            data1 += v[1]
            data2 += v[2]
            data3 += v[3]

        self.data = [data0, data1, data2, data3]
        self.label = label
        self.total = len(label)
        self.add_noise = add_noise

        print(self.total)


    def augment(self, data0, data1, data2, data3):
        data0 = data0.transpose(1, 0)
        data1 = data1.transpose(1, 0)
        data2 = data2.transpose(1, 0)
        data3 = data3.transpose(1, 0)
        idx = random.randint(0, len(data0) / 2)

        st_ed = random.randint(0, 1)

        # start
        if (st_ed == 0):
            for i in range(idx):
                data0[i][:] = data0[idx]
                data1[i][:] = data1[idx]
                data2[i][:] = data2[idx]
                data3[i][:] = data3[idx]
        # end
        else:
            idx += 1
            for i in range(1, idx):
                data0[-i][:] = data0[-idx]
                data1[-i][:] = data1[-idx]
                data2[-i][:] = data2[-idx]
                data3[-i][:] = data3[-idx]

        data0 = data0.transpose(1, 0)
        data1 = data1.transpose(1, 0)
        data2 = data2.transpose(1, 0)
        data3 = data3.transpose(1, 0)

        return data0, data1, data2, data3

    def __getitem__(self, index):
        # raw, no_pY, no_pX, no_all
        data0, data1, data2, data3 = self.data[0][index], self.data[1][index], self.data[2][index], self.data[3][index]
        label = self.label[index]

        if (self.add_noise and index % 2 == 0):
            data0, data1, data2, data3 = self.augment(data0, data1, data2, data3)

        data0 = np.array(data0, dtype="float32")
        data1 = np.array(data1, dtype="float32")
        data2 = np.array(data2, dtype="float32")
        data3 = np.array(data3, dtype="float32")

        # pA, pB, pY are not needed
        label = label[3:]
        return data0, data1, data2, data3, label

    def __len__(self):
        return self.total
