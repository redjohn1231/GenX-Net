####
#The following is only an example; 
#please organize the data in a similar format according to your own requirements.
####
class data:

    path_train = 'D:/(3)Experimental_data/(2)method_2/DATA/Dtrain/'
    path_val   = 'D:/(3)Experimental_data/(2)method_2/DATA/Dval/'
    path_test  = 'D:/(3)Experimental_data/(2)method_2/DATA/Dtest/'

    #训练集
    name = ['/(0)Deepfake(FF++)/',    '/(0)DeepFake(Celeb-DF)/',
            '/(1)PGGAN(DFFD)/',       '/(1)StyleGAN(seeprettyface)/',
            '/(2)Face2Face(FF++)/',
            '/(3)StarGAN(DFFD)/',
            '/(4)Midjourney(DiFF)/',  '/(4)SDXL(DiFF)/',
            '/(5)SDXL Refiner(DiFF)/','/(5)LoRA(DiFF)/',
            '/(6)DiffFace(DiFF)/',
            '/(7)Imagic(DiFF)/',      '/(7)CoDiff(DiFF)/',
            '/(8)FFHQ/',              '/(8)CelebA/']

    label = [0,0,
             1,1,
             2,
             3,
             4,4,
             5,5,
             6,
             7,7,
             8,8]

    path_1 = []
    for i in range(len(name)):
        path_1.append(path_train + name[i])

    path_2 = []
    for i in range(len(name)):
        path_2.append(path_val + name[i])
        
    name_1 = ['/(0)Deepfake(FF++)/',     '/(0)FaceSwap(FF++)/',      '/(0)DeepFake(Celeb-DF)/',     '/(0)FS(DFDC)/',
              '/(1)PGGAN(DFFD)/',        '/(1)StyleGAN(DFFD)/',      '/(1)StyleGAN(seeprettyface)/', 
              '/(2)Face2Face(FF++)/',    '/(2)NeuralTextures(FF++)/',
              '/(3)StarGAN(DFFD)/',      '/(3)FaceAPP(DFFD)/',
              '/(4)Midjourney(DiFF)/',   '/(4)SDXL(DiFF)/',          '/(4)FreeDoM_T(DiFF)/',        '/(4)HPS(DiFF)/',
              '/(5)SDXL Refiner(DiFF)/', '/(5)LoRA(DiFF)/',          '/(5)DeamBooth(DiFF)/',        '/(5)FreeDoM_I(DiFF)/',
              '/(6)DiffFace(DiFF)/',     '/(6)DCFace(DiFF)/',
              '/(7)Imagic(DiFF)/',       '/(7)CoDiff(DiFF)/',        '/(7)CycleDiff(DiFF)/',
              '/(8)FFHQ/',               '/(8)CelebA/',              '/(8)DFDC/']
   
    label_1 = [0,0,0,0,
               1,1,1,
               2,2,
               3,3,
               4,4,4,4,
               5,5,5,5,
               6,6,
               7,7,7,
               8,8,8]
    
    path_3 = []
    for i in range(len(name_1)):
        path_3.append(path_test + name_1[i])
        
class Hyperparameters:
    height = ...
    width = ...
    channel = ...
    batch_size_t = ...
    epochs = ...
    steps_t = ...
    lr = ...
    num = ...

    batch_size_v = ...
    steps_v = ...
    
    