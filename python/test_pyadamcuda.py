import torch
import pyadamcuda as ac
import numpy as np
import json

adamCuda = ac.AdamCuda()

# Input
proj_truth = np.array([
    706.4034576416011, 979.8575578523095,   1.0,  #// 14 right ankle
    737.4263076782222, 771.5612779450829,   1.0,  #// 13 right knee
    772.8809890747066, 558.8331790757592,   1.0,  #// 12 right hip
    905.8360519409175, 558.8331790757592,   1.0,  #// 6 left hip
    936.8589019775386, 784.8567796540673,   1.0,  #// 7 left knee
    941.2907333374019, 993.1530595612938,   1.0,  #// 8 left ankle
    604.4712371826167, 514.5148244691308,   1.0,  #// 11 right wrist
    666.5169372558589, 394.8552690339501,   1.0,  #// 10 right elbow
    755.1536483764644, 253.0365434480126,   1.0,  #// 9 right shoulder
    941.2907333374019, 257.46836717848134,  1.0,  #// 3 left shoulder
    1012.2001037597652, 408.15077074293447, 1.0,  #// 4 left elbow
    1074.2458038330074, 532.2421661210473,  1.0,  #// 5 left wrist
    848.2221918106075, 257.46836717848134,  1.0,  #// 0 neck,

    852.654026985168, 120.0814653230126,    1.0,  #// 1 nose
    874.8132057189937, 97.9222856355126,    1.0,  #// 15 left eye
    834.9266848564143, 97.9222856355126,    1.0,  #// 17 right eye
    901.404216766357, 120.0814653230126,    1.0,  #// 16 left ear
    803.9038352966304, 120.0814653230126,   1.0,  #// 18 right ear

    692.142, 1024.76,                       1.0,  #// right big toe
    671.498, 1021.69,                       1.0,  #// right small toe
    718.577, 1010.0,                        1.0,  #// right heel
    718.577, 1010.0,                        1.0,  #// right heel
    948.244, 1015.96,                       1.0,  #// left big toe
    971.827, 1010.01,                       1.0,  #// left small toe
    924.622, 1001.19,                       1.0,  #// left heel
    924.622, 1001.19,                       1.0,  #// left heel

    828.479, 105.578,                       0.0, #//36 (Face)
    842.987, 106.653,                       0.0, #//39
    868.241, 107.19,                        0.0, #//42
    884.36, 106.653,                        0.0, #//45
    832.778, 105.041,                       0.0, #//37
    838.151, 105.041,                       0.0, #//38
    873.077, 105.041,                       0.0, #//43
    880.062, 105.041,                       0.0, #//44
    832.778, 106.653,                       0.0, #//41
    838.151, 106.653,                       0.0, #//40
    873.614, 108.802,                       0.0, #//47
    880.062, 109.339,                       0.0, #//46
    854.808, 123.309,                       0.0, #//30
    847.286, 131.907,                       0.0, #//31
    850.509, 132.981,                       0.0, #//32
    854.808, 134.593,                       0.0, #//33
    859.107, 133.518,                       0.0, #//34
    863.405, 131.907,                       0.0, #//35
    851.047, 144.265,                       0.0, #//50
    854.808, 144.802,                       0.0, #//51
    859.107, 144.265,                       0.0, #//52
    838.688, 149.101,                       0.0, #//48
    871.465, 149.101,                       0.0, #//54
    854.808, 157.16,                        0.0, #//57
    850.509, 157.16,                        0.0, #//58
    859.107, 157.16,                        0.0, #//56
    854.808, 149.101,                       0.0, #//62
    854.808, 149.638,                       0.0, #//66
    850.509, 149.101,                       0.0, #//61
    859.107, 149.101,                       0.0, #//63
    850.509, 149.101,                       0.0, #//67
    859.107, 149.101,                       0.0, #//65
    867.704, 149.101,                       0.0, #//64
    841.912, 149.101,                       0.0, #//60
    846.211, 93.7569,                       0.0, #//21
    866.092, 94.2943,                       0.0, #//22
    832.778, 89.4584,                       0.0, #//19
    880.062, 91.6077,                       0.0, #//24
    855.345, 104.503,                       0.0, #//27
    0., 0.,                                 0.0,
    819.882, 94.2943,                       0.0, #//17
    892.958, 95.9062,                       0.0, #//26
    0., 0.,                                 0.0,
    0., 0.,                                 0.0,
    0., 0.,                                 0.0,
    0., 0.,                                 0.0,

    1077.0614013671875, 557.0326538085938,  0.5, #//22 (ADAM Joints)
    1079.4293212890625, 575.97607421875,    0.5, #//23
    1086.5330810546875, 593.735595703125,   0.5, #//24
    1092.4530029296875, 604.3912353515625,  0.5, #//25
    1105.0819091796875, 582.6852416992188,  0.5, #//26
    1112.9749755859375, 603.6019287109375,  0.5, #//27
    1118.500244140625, 620.1774291992188,   0.5, #//28
    1125.9986572265625, 630.8331298828125,  0.5, #//29
    1107.4498291015625, 581.1066284179688,  0.5, #//30
    1116.5269775390625, 602.0233154296875,  0.5, #//31
    1122.0521240234375, 614.2576293945312,  0.5, #//32
    1129.1558837890625, 623.334716796875,   0.5, #//33
    1105.0819091796875, 577.9494018554688,  0.5, #//34
    1112.1856689453125, 595.314208984375,   0.5, #//35
    1116.5269775390625, 607.5485229492188,  0.5, #//36
    0.0, 0.0,                               0.0, #// Some keypoint is not used for fitting if it is 0.0.
    1099.951416015625, 575.5814208984375,   0.5, #//38
    1104.2926025390625, 589.7890014648438,  0.5, #//39
    0.0, 0.0,                               0.0, #//40
    0.0, 0.0,                               0.0, #//41

    602.0302124023438, 528.9420776367188,   0.5,
    600.686767578125, 549.0941772460938,    0.5,
    593.9694213867188, 566.5592651367188,   0.5,
    589.0433349609375, 577.3070068359375,   0.5,
    576.9520874023438, 559.8419189453125,   0.5,
    568.8912963867188, 581.7852783203125,   0.5,
    562.1739501953125, 596.5634155273438,   0.5,
    555.9044189453125, 607.3112182617188,   0.5,
    575.1608276367188, 559.8419189453125,   0.5,
    564.8609008789062, 583.5765380859375,   0.5,
    559.4869995117188, 598.8025512695312,   0.5,
    553.2174682617188, 613.1329345703125,   0.5,
    578.2955932617188, 557.1549682617188,   0.5,
    569.7869262695312, 576.411376953125,    0.5,
    562.1739501953125, 595.219970703125,    0.5,
    554.1130981445312, 613.1329345703125,   0.5,
    582.7738037109375, 555.8115234375,      0.5,
    576.9520874023438, 567.9027099609375,   0.5,
    576.5042724609375, 580.4417724609375,   0.5,
    0.0, 0.0,                               0.0
], dtype=np.float32)
proj_truth = proj_truth.reshape((112,3))

pof_truth = np.array([
    -0.24057312309741974, 1.0784096717834473, -0.07078318297863007,     150, #// 0 neck : r hip
    -0.2039204090833664, 1.0271286964416504, -0.2153000682592392,       150, #// 1 r hip : r knee
    -0.1532621532678604, 1.0642367601394653, 0.19312481582164764,       150, #// 2 r knee : r ankle
    0.18657980859279633, 1.1015098094940186, -0.04150465503334999,      150, #// 3 neck : l hip
    0.15125027298927307, 1.0648096799850464, -0.12637579441070557,      150, #// 4 l hip : l knee
    0.007774390745908022, 1.1358691453933716, 0.10240084677934647,      150, #// 5 l knee : l ankle
    -1.0797539949417114, 0.007119467481970787, -0.010265235789120197,   150, #// 6 neck : r shoulder
    -0.5971511602401733, 0.8917768597602844, -0.07271111756563187,      150, #// 7 r shouder : r elbow
    -0.35578832030296326, 0.8548061847686768, -0.6225680708885193,      150, #// 8 r elbow : r wrist
    1.094591498374939, 0.052694521844387054, 0.05548678711056709,       150, #// 9 neck : l shoulder
    0.476468950510025, 0.9687181711196899, 0.17159108817577362,         150, #// 10 l shoulder : l elbow
    0.4357653558254242, 0.9652106761932373, -0.16269558668136597,       150, #// 11 l elbow : l wrist
    0.02512143738567829, -0.9025842547416687, -0.6771435141563416,      150, #// 12 neck : nose

    -0.12957115471363068, 0.5362922549247742, -0.5084066987037659,      50, #//Left Hand POF
    0.02298053540289402, 0.6550264954566956, -0.4624072313308716,       50,
    0.2772134244441986, 0.6872079968452454, -0.3035981357097626,        50,
    0.2853514850139618, 0.6174838542938232, -0.47691935300827026,       50,
    0.4265272915363312, 0.648213267326355, -0.35114216804504395,        50,
    0.2772155702114105, 0.6823596358299255, -0.00672254990786314,       50,
    0.16775481402873993, 0.45171043276786804, -0.10401373356580734,     50,
    0.1290237009525299, 0.36327266693115234, -0.14357012510299683,      50,
    0.5474904179573059, 0.7110057473182678, -0.1462286114692688,        50,
    0.29382067918777466, 0.7006406784057617, -0.07686197012662888,      50,
    0.2269606739282608, 0.6766166090965271, -0.055553138256073,         50,
    0.19569139182567596, 0.4314614236354828, -0.07351892441511154,      50,
    0.47968608140945435, 0.6795504093170166, 0.07272634655237198,       50,
    0.26544415950775146, 0.6970440149307251, 0.045050378888845444,      50,
    0.11786217987537384, 0.5371779799461365, 0.00133417802862823,       50,
    0.0, 0.0, 0.0,                                                      0,
    0.36390799283981323, 0.5406317114830017, 0.2263338416814804,        50,
    0.10784879326820374, 0.3973357379436493, 0.054951347410678864,      50,
    0.0, 0.0, 0.0,                                                      0,
    0.0, 0.0, 0.0,                                                      0,

    0.08721618354320526, 0.6381232738494873, -0.5412018895149231,       50, #//Right Hand POF
    0.02870943583548069, 0.7755728363990784, -0.45594194531440735,      50,
    -0.29710543155670166, 0.780907392501831, -0.3162669837474823,       50,
    -0.25263258814811707, 0.6233384609222412, -0.4532473683357239,      50,
    -0.37454482913017273, 0.7238391041755676, -0.36881333589553833,     50,
    -0.2889643907546997, 0.6873354315757751, -0.041378363966941833,     50,
    -0.19252508878707886, 0.4793771505355835, -0.13254694640636444,     50,
    -0.17174892127513885, 0.41723883152008057, -0.23752754926681519,    50,
    -0.4803922772407532, 0.8099092245101929, -0.18307334184646606,      50,
    -0.3340895473957062, 0.7537098526954651, -0.19354833662509918,      50,
    -0.20890241861343384, 0.6349567770957947, -0.2198496013879776,      50,
    -0.2072679102420807, 0.5821881890296936, -0.20913024246692657,      50,
    -0.3974188566207886, 0.7915560007095337, 0.03410215303301811,       50,
    -0.23765744268894196, 0.7553504705429077, -0.07207278907299042,     50,
    -0.18477830290794373, 0.5315693616867065, -0.1492283046245575,      50,
    -0.14372003078460693, 0.399129182100296, -0.13902486860752106,      50,
    -0.30553188920021057, 0.6624711751937866, 0.19307294487953186,      50,
    -0.11071541905403137, 0.39049795269966125, -1.7697499060886912e-05, 50,
    -0.05220204219222069, 0.17173926532268524, -0.046017929911613464,   50,
    0.0, 0.0, 0.0,                                                      0
], dtype=np.float32)
pof_truth = pof_truth.reshape((53,4))

calib = np.array([2000., 0., 960., 0., 2000., 540., 0., 0., 1.], dtype=np.float32)

proj_truth_tensor = torch.tensor(proj_truth).cuda()
pof_truth_tensor = torch.tensor(pof_truth).cuda()
calib_tensor = torch.tensor(calib).cuda()

adamCuda.proj_truth_tensor = proj_truth_tensor
adamCuda.pof_truth_tensor = pof_truth_tensor
adamCuda.calib_tensor = calib_tensor

# Initial Variables
t_tensor = torch.tensor(np.array([0,0,200], dtype=np.float32)).cuda()
eulers_tensor = torch.tensor(np.zeros((62,3), dtype=np.float32)).cuda()
bodyshape_tensor = torch.tensor(np.zeros((30,1), dtype=np.float32)).cuda()
faceshape_tensor = torch.tensor(np.zeros((200,1), dtype=np.float32)).cuda()

adamCuda.run(t_tensor, eulers_tensor, bodyshape_tensor, faceshape_tensor, True) # First time is slow
adamCuda.run(t_tensor, eulers_tensor, bodyshape_tensor, faceshape_tensor, True)

#adamCuda.r_tensor
#adamCuda.drdt_tensor
#adamCuda.drdP_tensor
#adamCuda.drdc_tensor
#adamCuda.drdf_tensor
