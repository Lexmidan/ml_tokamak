import LHmode_classifier as LH
import alt_models_training as amtr
import Cross_validation as cval

import torch
import torchvision

from torchvision.models.resnet import ResNet50_Weights, ResNet34_Weights, ResNet101_Weights, ResNet152_Weights, ResNet18_Weights

from importlib import reload
import traceback
if __name__ == "__main__":

    try:
        amtr.train_and_test_alt_model(signal_name = 'mc',
                                    architecture = 'Simple1DCNN',
                                    signal_window = 320,
                                    sampling_freq = 300,
                                    batch_size = 256,
                                    num_workers = 32,
                                    num_epochs = 16,
                                    dpoints_in_future = 160,
                                    learning_rate_max = 0.01,
                                    num_classes=3,
                                    weight_decay=0.1,
                                    comment_for_model_name = f'mirnov coil, 160 dpoints_in_future, 320 window, 300 sampling_freq',
                                    exponential_elm_decay=True,
                                    use_ELMs=True,
                                    no_L_mode=False)
    except Exception as e:
        with open('exception.log', 'w') as f:
            f.write(str(e) + '\n')
    
    try:
        amtr.train_and_test_alt_model(signal_name = 'h_alpha',
                                    architecture = 'Simple1DCNN',
                                    signal_window = 320,
                                    sampling_freq = 300,
                                    batch_size = 256,
                                    num_workers = 32,
                                    num_epochs = 16,
                                    dpoints_in_future = 160,
                                    learning_rate_max = 0.01,
                                    num_classes=3,
                                    weight_decay=0.1,
                                    comment_for_model_name = f'h_alpha, 160 dpoints_in_future, 320 window, 300 sampling_freq',
                                    exponential_elm_decay=True,
                                    use_ELMs=True,
                                    no_L_mode=False)
    except Exception as e:
        with open('exception.log', 'w') as f:
            f.write(str(e) + '\n')

    try:
        amtr.train_and_test_alt_model(signal_name = 'mc_h_alpha',
                                    architecture = 'Simple1DCNN',
                                    signal_window = 320,
                                    sampling_freq = 300,
                                    batch_size = 256,
                                    num_workers = 32,
                                    num_epochs = 16,
                                    dpoints_in_future = 160,
                                    learning_rate_max = 0.01,
                                    num_classes=3,
                                    weight_decay=0.1,
                                    comment_for_model_name = f'mc_h_alpha, 160 dpoints_in_future, 320 window, 300 sampling_freq',
                                    exponential_elm_decay=True,
                                    use_ELMs=True,
                                    no_L_mode=False)
    except Exception as e:
        with open('exception.log', 'w') as f:
            f.write(str(e) + '\n')


    # try:
    #     LH.train_and_test_ris_model(ris_option = 'both',
    #                                 pretrained_model = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1),
    #                                 num_workers = 32,
    #                                 num_epochs_for_fc = 10,
    #                                 num_epochs_for_all_layers = 16,
    #                                 num_classes = 3,
    #                                 batch_size = 32,
    #                                 learning_rate_min = 0.001,
    #                                 learning_rate_max = 0.01,
    #                                 comment_for_model_name = f', ResNet34',
    #                                 random_seed = 42,
    #                                 augmentation = False,
    #                                 test_df_contains_val_df=True,
    #                                 test_run = False,
    #                                 exponential_elm_decay=False,
    #                                 grayscale=False,
    #                                 weight_decay=1e-4,
    #                                 data_frac=1.0)
    
    # except Exception as e:
    #     with open('./runs/exception.log', 'w') as f:
    #         f.write(str(e) + '\n')

    # try:
    #     LH.train_and_test_ris_model(ris_option = 'RIS1',
    #                                 pretrained_model = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1),
    #                                 num_workers = 32,
    #                                 num_epochs_for_fc = 10,
    #                                 num_epochs_for_all_layers = 16,
    #                                 num_classes = 3,
    #                                 batch_size = 32,
    #                                 learning_rate_min = 0.001,
    #                                 learning_rate_max = 0.01,
    #                                 comment_for_model_name = f', ResNet34',
    #                                 random_seed = 42,
    #                                 augmentation = False,
    #                                 test_df_contains_val_df=True,
    #                                 test_run = False,
    #                                 exponential_elm_decay=False,
    #                                 grayscale=False,
    #                                 weight_decay=1e-4,
    #                                 data_frac=1.0)
    
    # except Exception as e:
    #     with open('./runs/exception.log', 'w') as f:
    #         f.write(str(e) + '\n')
    
    # try:
    #     LH.train_and_test_ris_model(ris_option = 'RIS2',
    #                                 pretrained_model = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1),
    #                                 num_workers = 32,
    #                                 num_epochs_for_fc = 10,
    #                                 num_epochs_for_all_layers = 16,
    #                                 num_classes = 3,
    #                                 batch_size = 32,
    #                                 learning_rate_min = 0.001,
    #                                 learning_rate_max = 0.01,
    #                                 comment_for_model_name = f', ResNet34',
    #                                 random_seed = 42,
    #                                 augmentation = False,
    #                                 test_df_contains_val_df=True,
    #                                 test_run = False,
    #                                 exponential_elm_decay=False,
    #                                 grayscale=False,
    #                                 weight_decay=1e-4,
    #                                 data_frac=1.0)
    
    # except Exception as e:
    #     with open('./runs/exception.log', 'w') as f:
    #         f.write(str(e) + '\n')

    # torch.cuda.empty_cache()

    # try:
    #     LH.train_and_test_ris_model(ris_option = 'both',
    #                                 pretrained_model = torchvision.models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1),
    #                                 num_workers = 32,
    #                                 num_epochs_for_fc = 10,
    #                                 num_epochs_for_all_layers = 16,
    #                                 num_classes = 3,
    #                                 batch_size = 32,
    #                                 learning_rate_min = 0.001,
    #                                 learning_rate_max = 0.01,
    #                                 comment_for_model_name = f', ResNet101',
    #                                 random_seed = 42,
    #                                 augmentation = False,
    #                                 test_df_contains_val_df=True,
    #                                 test_run = False,
    #                                 exponential_elm_decay=False,
    #                                 grayscale=False,
    #                                 weight_decay=1e-4,
    #                                 data_frac=1.0)
    
    # except Exception as e:
    #     with open('./runs/exception.log', 'w') as f:
    #         f.write(str(e) + '\n')
    #         traceback.print_exc(file=f)

