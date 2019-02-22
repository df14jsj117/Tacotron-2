import numpy as np
import tensorflow as tf

# Default hyperparameters
hparams = tf.contrib.training.HParams(
	# Comma-separated list of cleaners to run on text prior to training and eval. For non-English
	# text, you may want to use "basic_cleaners" or "transliteration_cleaners".
	cleaners='basic_cleaners',

	#Hardware setup (TODO: multi-GPU parallel tacotron training)
	use_all_gpus = False, #Whether to use all GPU resources. If True, total number of available gpus will override num_gpus.
	num_gpus = 1, #Determines the number of gpus in use
	###########################################################################################################################################

	#Audio
	num_mels = 160, #Number of mel-spectrogram channels and local conditioning dimensionality
	num_freq = 1025, # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
	rescale = True, #Whether to rescale audio prior to preprocessing
	rescaling_max = 0.999, #Rescaling value
	trim_silence = True, #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
	clip_mels_length = True, #For cases of OOM (Not really recommended, working on a workaround)
	max_mel_frames = 900,  #Only relevant when clip_mels_length = True

	# Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
	# It's preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
	# Does not work if n_ffit is not multiple of hop_size!!
	use_lws=False,
	silence_threshold=2, #silence threshold used for sound trimming for wavenet preprocessing

	#Mel spectrogram
	n_fft = 2048, #Extra window size is filled with 0 paddings to match this parameter
	hop_size = 256, #For 22050Hz, 275 ~= 12.5 ms
	win_size = 1024, #For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft)
	sample_rate = 16000, #22050 Hz (corresponding to ljspeech dataset)
	frame_shift_ms = None,
	preemphasis = 0.97, # preemphasis coefficient

	#M-AILABS (and other datasets) trim params
	trim_fft_size = 512,
	trim_hop_size = 128,
	trim_top_db = 60,

	#Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization = True,
	allow_clipping_in_normalization = False, #Only relevant if mel_normalization = True
	symmetric_mels = True, #Whether to scale the data to be symmetric around 0
	max_abs_value = 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max]
	normalize_for_wavenet = True, #whether to rescale to [0, 1] for wavenet.

	#Limits
	min_level_db = -120,
	ref_level_db = 20,
	fmin = 55, #Set this to 75 if your speaker is male! if female, 125 should help taking off noise. (To test depending on dataset)
	fmax = 3000,

	#Griffin Lim
	power = 1.2,
	griffin_lim_iters = 60,
	###########################################################################################################################################

	#Tacotron
	outputs_per_step = 2, #number of frames to generate at each decoding step (speeds up computation and allows for higher batch size)
	stop_at_any = True, #Determines whether the decoder should stop when predicting <stop> to any frame or to all of them

	embedding_dim = 512, #dimension of embedding space

	enc_conv_num_layers = 3, #number of encoder convolutional layers
	enc_conv_kernel_size = (5, ), #size of encoder convolution filters for each layer
	enc_conv_channels = 512, #number of encoder convolutions filters for each layer
	encoder_lstm_units = 256, #number of lstm units for each direction (forward and backward)

	smoothing = False, #Whether to smooth the attention normalization function
	attention_dim = 128, #dimension of attention space
	attention_filters = 32, #number of attention convolution filters
	attention_kernel = (31, ), #kernel size of attention convolution
	cumulative_weights = True, #Whether to cumulate (sum) all previous attention weights or simply feed previous weights (Recommended: True)

	prenet_layers = [256, 256], #number of layers and number of units of prenet
	decoder_layers = 2, #number of decoder lstm layers
	decoder_lstm_units = 1024, #number of decoder lstm units on each layer
	max_iters = 1000, #Max decoder steps during inference (Just for safety from infinite loop cases)

	postnet_num_layers = 5, #number of postnet convolutional layers
	postnet_kernel_size = (5, ), #size of postnet convolution filters for each layer
	postnet_channels = 512, #number of postnet convolution filters for each layer

	mask_encoder = False, #whether to mask encoder padding while computing attention
	mask_decoder = False, #Whether to use loss mask for padded sequences (if False, <stop_token> loss function will not be weighted, else recommended pos_weight = 20)

	cross_entropy_pos_weight = 1, #Use class weights to reduce the stop token classes imbalance (by adding more penalty on False Negatives (FN)) (1 = disabled)
	predict_linear = True, #Whether to add a post-processing network to the Tacotron to predict linear spectrograms (True mode Not tested!!)
	###########################################################################################################################################


	#Wavenet
	# Input type:
	# 1. raw [-1, 1]
	# 2. mulaw [-1, 1]
	# 3. mulaw-quantize [0, mu]
	# If input_type is raw or mulaw, network assumes scalar input and
	# discretized mixture of logistic distributions output, otherwise one-hot
	# input and softmax output are assumed.
	input_type="raw",
	quantize_channels=2 ** 16,  # 65536 (16-bit) (raw) or 256 (8-bit) (mulaw or mulaw-quantize) // number of classes = 256 <=> mu = 255

	log_scale_min=float(np.log(1e-14)), #Mixture of logistic distributions minimal log scale
	log_scale_min_gauss = float(np.log(1e-7)), #Gaussian distribution minimal allowed log scale

	#To use Gaussian distribution as output distribution instead of mixture of logistics, set "out_channels = 2" instead of "out_channels = 10 * 3". (UNDER TEST)
	out_channels = 2, #This should be equal to quantize channels when input type is 'mulaw-quantize' else: num_distributions * 3 (prob, mean, log_scale).
	layers = 30, #Number of dilated convolutions (Default: Simplified Wavenet of Tacotron-2 paper)
	stacks = 3, #Number of dilated convolution stacks (Default: Simplified Wavenet of Tacotron-2 paper)
	residual_channels = 512,
	gate_channels = 512, #split in 2 in gated convolutions
	skip_out_channels = 256,
	kernel_size = 3,

	cin_channels = 80, #Set this to -1 to disable local conditioning, else it must be equal to num_mels!!
	upsample_conditional_features = True, #Whether to repeat conditional features or upsample them (The latter is recommended)
	upsample_scales = [15, 20], #prod(upsample_scales) should be equal to hop_size
	freq_axis_kernel_size = 3,
	leaky_alpha = 0.4,

	gin_channels = -1, #Set this to -1 to disable global conditioning, Only used for multi speaker dataset. It defines the depth of the embeddings (Recommended: 16)
	use_speaker_embedding = True, #whether to make a speaker embedding
	n_speakers = 5, #number of speakers (rows of the embedding)

	use_bias = True, #Whether to use bias in convolutional layers of the Wavenet

	max_time_sec = None,
	max_time_steps = 13000, #Max time steps in audio used to train wavenet (decrease to save memory) (Recommend: 8000 on modest GPUs, 13000 on stronger ones)
	###########################################################################################################################################

	#Tacotron Training
	tacotron_random_seed = 5339, #Determines initial graph and operations (i.e: model) random state for reproducibility
	tacotron_swap_with_cpu = False, #Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause major slowdowns! Only use when critical!)

	tacotron_batch_size = 32, #number of training samples on each training steps
	tacotron_reg_weight = 1e-6, #regularization weight (for L2 regularization)
	tacotron_scale_regularization = True, #Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is high and biasing the model)

	tacotron_test_size = None, #% of data to keep as test data, if None, tacotron_test_batches must be not None
	tacotron_test_batches = 32, #number of test batches (For Ljspeech: 10% ~= 41 batches of 32 samples)
	tacotron_data_random_state=1234, #random state for train test split repeatability

	#Usually your GPU can handle 16x tacotron_batch_size during synthesis for the same memory amount during training (because no gradients to keep and ops to register for backprop)
	tacotron_synthesis_batch_size = 32 * 16, #This ensures GTA synthesis goes up to 40x faster than one sample at a time and uses 100% of your GPU computation power.

	tacotron_decay_learning_rate = True, #boolean, determines if the learning rate will follow an exponential decay
	tacotron_start_decay = 50000, #Step at which learning decay starts
	tacotron_decay_steps = 50000, #Determines the learning rate decay slope (UNDER TEST)
	tacotron_decay_rate = 0.4, #learning rate decay rate (UNDER TEST)
	tacotron_initial_learning_rate = 1e-3, #starting learning rate
	tacotron_final_learning_rate = 1e-5, #minimal learning rate

	tacotron_adam_beta1 = 0.9, #AdamOptimizer beta1 parameter
	tacotron_adam_beta2 = 0.999, #AdamOptimizer beta2 parameter
	tacotron_adam_epsilon = 1e-6, #AdamOptimizer beta3 parameter

	tacotron_zoneout_rate = 0.1, #zoneout rate for all LSTM cells in the network
	tacotron_dropout_rate = 0.5, #dropout rate for all convolutional layers + prenet

	tacotron_clip_gradients = True, #whether to clip gradients
	natural_eval = False, #Whether to use 100% natural eval (to evaluate Curriculum Learning performance) or with same teacher-forcing ratio as in training (just for overfit)

	#Decoder RNN learning can take be done in one of two ways:
	#	Teacher Forcing: vanilla teacher forcing (usually with ratio = 1). mode='constant'
	#	Curriculum Learning Scheme: From Teacher-Forcing to sampling from previous outputs is function of global step. (teacher forcing ratio decay) mode='scheduled'
	#The second approach is inspired by:
	#Bengio et al. 2015: Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.
	#Can be found under: https://arxiv.org/pdf/1506.03099.pdf
	tacotron_teacher_forcing_mode = 'constant', #Can be ('constant' or 'scheduled'). 'scheduled' mode applies a cosine teacher forcing ratio decay. (Preference: scheduled)
	tacotron_teacher_forcing_ratio = 1., #Value from [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder inputs, Only relevant if mode='constant'
	tacotron_teacher_forcing_init_ratio = 1., #initial teacher forcing ratio. Relevant if mode='scheduled'
	tacotron_teacher_forcing_final_ratio = 0., #final teacher forcing ratio. Relevant if mode='scheduled'
	tacotron_teacher_forcing_start_decay = 20000, #starting point of teacher forcing ratio decay. Relevant if mode='scheduled'
	tacotron_teacher_forcing_decay_steps = 280000, #Determines the teacher forcing ratio decay slope. Relevant if mode='scheduled'
	tacotron_teacher_forcing_decay_alpha = 0., #teacher forcing ratio decay rate. Relevant if mode='scheduled'
	###########################################################################################################################################

	#Wavenet Training
	wavenet_random_seed = 5339, # S=5, E=3, D=9 :)
	wavenet_swap_with_cpu = False, #Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause major slowdowns! Only use when critical!)

	wavenet_batch_size = 4, #batch size used to train wavenet.
	wavenet_test_size = 0.0441, #% of data to keep as test data, if None, wavenet_test_batches must be not None
	wavenet_test_batches = None, #number of test batches.
	wavenet_data_random_state = 1234, #random state for train test split repeatability

	#During synthesis, there is no max_time_steps limitation so the model can sample much longer audio than 8k(or 13k) steps. (Audio can go up to 500k steps, equivalent to ~21sec on 24kHz)
	#Usually your GPU can handle 1x~2x wavenet_batch_size during synthesis for the same memory amount during training (because no gradients to keep and ops to register for backprop)
	wavenet_synthesis_batch_size = 4 * 2, #This ensure that wavenet synthesis goes up to 4x~8x faster when synthesizing multiple sentences. Watch out for OOM with long audios.

	wavenet_learning_rate = 1e-3,
	wavenet_adam_beta1 = 0.9,
	wavenet_adam_beta2 = 0.999,
	wavenet_adam_epsilon = 1e-8,

	wavenet_ema_decay = 0.9999, #decay rate of exponential moving average

	wavenet_dropout = 0.05, #drop rate of wavenet layers
	train_with_GTA = True, #Whether to use GTA mels to train WaveNet instead of ground truth mels.
	###########################################################################################################################################

	#Eval sentences (if no eval file was specified, these sentences are used for eval)
	sentences = [
	# "yu2 jian4 jun1 : wei4 mei3 ge4 you3 cai2 neng2 de ren2 ti2 gong1 ping2 tai2 .",
	# "ta1 shi4 yin1 pin2 ling3 yu4 de tao2 bao3 tian1 mao1 , zai4 zhe4 ge4 ping2 tai2 shang4, ",
	# "mei3 ge4 nei4 rong2 sheng1 chan3 zhe3 dou1 ke3 yi3 hen3 fang1 bian4 de shi1 xian4 zi4 wo3 jia4 zhi2 , geng4 duo1 de ren2 yong1 you3 wei1 chuang4 ye4 de ji1 hui4 .",
	# "zui4 jin4 xi3 ma3 la1 ya3 de bao4 guang1 lv4 you3 dian3 gao1 , ren4 xing4 shai4 chu1 yi1 dian3 qi1 yi4 yuan2 de zhang4 hu4 yu2 e2 de jie2 tu2 ,",
	# "rang4 ye4 nei4 ye4 wai4 dou1 hen3 jing1 tan4 : yi2 ge4 zuo4 yin1 pin2 de , ju1 ran2 you3 zhe4 me duo1 qian2 ?",
	# "ji4 zhe3 cha2 dao4 , wang3 shang4 dui4 xi3 ma3 la1 ya3 de jie4 shao4 shi4 ,",
	# "xun4 su4 cheng2 zhang3 wei4 zhong1 guo2 zui4 da4 de yin1 pin2 fen1 xiang3 ping2 tai2 , mu4 qian2 yi3 yong1 you3 liang3 yi4 yong4 hu4 , qi3 ye4 zong3 gu1 zhi2 chao1 guo4 san1 shi2 yi4 yuan2 ren2 min2 bi4 .",
	# "jin4 ri4 , ji4 zhe3 zai4 shang4 hai3 zhang1 jiang1 gao1 ke1 ji4 yuan2 qu1 de xi3 ma3 la1 ya3 ji1 di4 zhuan1 fang3 le yu2 jian4 jun1 .",
	# "ta1 men dou1 shi4 han3 ta1 lao3 yu2 de , bu4 guo4 hou4 lai2 ji4 zhe3 wen4 guo4 ta1 de nian2 ling2 , qi2 shi2 cai2 yi1 jiu3 qi1 qi1 nian2 de .",
	# "ji4 zhe3 liao3 jie3 dao4 , xi3 ma3 la1 ya3 cai3 qu3 bu4 duo1 jian4 de lian2 xi2 mo2 shi4 , ling4 yi1 wei4 jiu4 shi4 chen2 xiao3 yu3 ,",
	# "liang3 ren2 qi4 zhi4 hun4 da1 , you3 dian3 nan2 zhu3 wai4 nv3 zhu3 nei4 de yi4 si1 ,",
	# "bu4 guo4 ta1 men zhi3 shi4 da1 dang4 , bu2 shi4 chang2 jian4 de fu1 qi1 dang4 mo2 shi4 . yong4 yu2 jian4 jun1 de hua4 lai2 shuo1 , zhe4 ge4 mo2 shi4 ye3 bu4 chang2 jian4 .",

	# "huan2 qiu2 wang3 bao4 dao4 , e2 luo2 si1 wei4 xing1 wang3 shi2 yi1 ri4 bao4 dao4 cheng1 , ji4 nian4 di4 yi2 ci4 shi4 jie4 da4 zhan4 jie2 shu4 yi4 bai3 zhou1 nian2 qing4 zhu4 dian3 li3 zai4 ba1 li2 ju3 xing2 , , ",
	# "e2 luo2 si1 zong3 tong3 pu3 jing1 he2 mei3 guo2 zong3 tong3 te4 lang3 pu3 zai4 ba1 li2 kai3 xuan2 men2 jian4 mian4 shi2 wo4 shou3 zhi4 yi4 . .",
	# "pu3 jing1 biao3 shi4 , tong2 mei3 guo2 zong3 tong3 te4 lang3 pu3 jin4 xing2 le hen3 hao3 de jiao1 liu2 . . ",
	# "e2 luo2 si1 zong3 tong3 zhu4 shou3 you2 li3 , wu1 sha1 ke1 fu1 biao3 shi4 , fa3 guo2 fang1 mian4 zhi2 yi4 yao1 qiu2 bu2 yao4 zai4 ba1 li2 ju3 xing2 ji4 nian4 huo2 dong4 qi1 jian1 ju3 xing2 e2 mei3 liang3 guo2 zong3 tong3 de dan1 du2 hui4 wu4 . . ",
	# "da2 cheng2 le xie2 yi4 , wo3 men yi3 jing1 kai1 shi3 xie2 tiao2 e2 luo2 si1 he2 mei3 guo2 zong3 tong3 hui4 wu4 de shi2 jian1 , dan4 hou4 lai2 wo3 men kao3 lv4 dao4 le fa3 guo2 tong2 hang2 men de dan1 you1 he2 guan1 qie4 . wu1 sha1 ke1 fu1 shuo1 . . ",
	# "yin1 ci3 , wo3 men yu3 mei3 guo2 dai4 biao3 men yi4 qi3 jin4 xing2 le tao3 lun4 , jue2 ding4 zai4 bu4 yi2 nuo4 si1 ai4 li4 si1 feng1 hui4 shang4 jin4 xing2 nei4 rong2 geng4 feng1 fu4 de dui4 hua4 . .",
	# "bao4 dao4 cheng1 , pu3 jing1 he2 te4 lang3 pu3 zai4 ai4 li4 she4 gong1 wu3 can1 hui4 shang4 de zuo4 wei4 an1 pai2 zai4 zui4 hou4 yi1 fen1 zhong1 jin4 xing2 le tiao2 zheng3 , dan4 zhe4 bing4 bu4 fang2 ai4 ta1 men jiao1 tan2 . . ",
	# "sui1 ran2 dong1 dao4 zhu3 fa3 guo2 dui4 ta1 men zai4 ba1 li2 de hui4 wu4 biao3 shi4 fan3 dui4 , dan4 e2 mei3 ling3 dao3 ren2 reng2 ran2 biao3 shi4 , ta1 men xi1 wang4 zai4 ai4 li4 she4 gong1 de gong1 zuo4 wu3 can1 shang4 hui4 mian4 . . ",
	# "chu1 bu4 zuo4 wei4 biao3 xian3 shi4 , te4 lang3 pu3 bei4 an1 pai2 zai4 pu3 jing1 pang2 bian1 , dan4 zai4 sui2 hou4 jin4 xing2 de gong1 zuo4 wu3 can1 qi1 jian1 , zuo4 wei4 an1 pai2 xian3 ran2 yi3 jing1 fa1 sheng1 le bian4 hua4 . . ",
	# "cong2 zhao4 pian1 lai2 kan4 , pu3 jing1 dang1 shi2 zheng4 quan2 shen2 guan4 zhu4 de yu3 lian2 he2 guo2 mi4 shu1 chang2 gu3 te4 lei2 si1 jiao1 tan2 , ou1 meng2 wei3 yuan2 hui4 zhu3 xi2 rong2 ke4 zuo4 zai4 pu3 jing1 de you4 bian1 , , ",
	# "er2 te4 lang3 pu3 ze2 zuo4 zai4 ma3 ke4 long2 pang2 bian1 , ma3 ke4 long2 de you4 bian1 ze2 shi4 de2 guo2 zong3 li3 mo4 ke4 er3 . . ",
	# "ci3 qian2 , pu3 jing1 zai4 fang3 wen4 ba1 li2 qi1 jian1 biao3 shi4 , ta1 bu4 pai2 chu2 yu3 te4 lang3 pu3 zai4 gong1 zuo4 wu3 can1 shi2 jin4 xing2 jiao1 liu2 . . ",
	# "pu3 jing1 zai4 fa3 guo2 pin2 dao4 de jie2 mu4 zhong1 hui2 da2 shi4 fou3 yi3 tong2 te4 lang3 pu3 jin4 xing2 jiao1 liu2 de wen4 ti2 shi2 biao3 shi4 zan4 shi2 mei2 you3 , wo3 men zhi3 da3 le ge4 zhao1 hu1 . . ",
	# "yi2 shi4 yi3 zhe4 yang4 de fang1 shi4 jin4 xing2 , wo3 men wu2 fa3 zai4 na4 li3 jin4 xing2 jiao1 liu2 , wo3 men guan1 kan4 le fa1 sheng1 de shi4 qing2 . . ",
	# "dan4 xian4 zai4 hui4 you3 gong1 zuo4 wu3 can1 , ye3 xu3 zai4 na4 li3 wo3 men hui4 jin4 xing2 jie1 chu4 . . ",
	# "dan4 shi4 , wu2 lun4 ru2 he2 , wo3 men shang1 ding4 , wo3 men zai4 zhe4 li3 bu4 hui4 wei2 fan3 zhu3 ban4 guo2 de gong1 zuo4 an1 pai2 , gen1 ju4 ta1 men de yao1 qiu2 , ",
	# "wo3 men bu4 hui4 zai4 zhe4 li3 zu3 zhi1 ren4 he2 hui4 mian4 , er2 shi4 ke3 neng2 hui4 zai4 G er4 ling2 qi1 jian1 huo4 zai4 ci3 zhi1 hou4 ju3 xing2 hui4 mian4 . . ",
	# "pu3 jing1 hai2 biao3 shi4 , e2 luo2 si1 zhun3 bei4 tong2 mei3 guo2 jin4 xing2 dui4 hua4 , fan3 zheng4 bu2 shi4 mo4 si1 ke1 yao4 tui4 chu1 zhong1 dao3 tiao2 yue1 . . ",

	"huan2 qiu2 wang3 bao4 dao4 .",
	"e2 luo2 si1 wei4 xing1 wang3 shi2 yi1 ri4 bao4 dao4 cheng1 .",
	"ji4 nian4 di4 yi2 ci4 shi4 jie4 da4 zhan4 jie2 shu4 yi4 bai3 zhou1 nian2 qing4 zhu4 dian3 li3 ,",
	"zai4 ba1 li2 ju3 xing2 .",
	"e2 luo2 si1 zong3 tong3 pu3 jing1 ",
	"he2 mei3 guo2 zong3 tong3 te4 lang3 pu3 ",
	"zai4 ba1 li2 kai3 xuan2 men2 jian4 mian4 shi2 ",
	"wo4 shou3 zhi4 yi4 .",
	"pu3 jing1 biao3 shi4 .",
	"tong2 mei3 guo2 zong3 tong3 te4 lang3 pu3 ,",
	"jin4 xing2 le hen3 hao3 de jiao1 liu2 .",
	"e2 luo2 si1 zong3 tong3 zhu4 shou3 ",
	"you2 li3 wu1 sha1 ke1 fu1 biao3 shi4 .",
	"fa3 guo2 fang1 mian4 zhi2 yi4 yao1 qiu2 ,",
	"bu2 yao4 zai4 ba1 li2 ju3 xing2 ji4 nian4 huo2 dong4 qi1 jian1 ." ,
	"ju3 xing2 e2 mei3 liang3 guo2 zong3 tong3 de dan1 du2 hui4 wu4 .",
	"da2 cheng2 le xie2 yi4 .",
	"wo3 men yi3 jing1 kai1 shi3 xie2 tiao2 ,",
	"e2 luo2 si1 he2 mei3 guo2 zong3 tong3 hui4 wu4 de shi2 jian1 .",
	"dan4 hou4 lai2 ,",
	"wo3 men kao3 lv4 dao4 le fa3 guo2 tong2 hang2 men de dan1 you1 he2 guan1 qie4 .",
	"wu1 sha1 ke1 fu1 shuo1 .",
	"yin1 ci3 ,",
	"wo3 men yu3 mei3 guo2 dai4 biao3 men yi4 qi3 ,",
	"jin4 xing2 le tao3 lun4 .",
	"jue2 ding4 ,",
	"zai4 bu4 yi2 nuo4 si1 ai4 li4 si1 feng1 hui4 shang4 ,",
	"jin4 xing2 nei4 rong2 geng4 feng1 fu4 de dui4 hua4 .",

	"bao4 dao4 cheng1 .",
	"pu3 jing1 he2 te4 lang3 pu3 zai4 ai4 li4 she4 gong1 wu3 can1 hui4 shang4 de zuo4 wei4 an1 pai2 zai4 zui4 hou4 yi1 fen1 zhong1 jin4 xing2 le tiao2 zheng3 .",
	"dan4 zhe4 bing4 bu4 fang2 ai4 ta1 men jiao1 tan2 .",
	"sui1 ran2 dong1 dao4 zhu3 fa3 guo2 dui4 ta1 men zai4 ba1 li2 de hui4 wu4 biao3 shi4 fan3 dui4 .",
	"dan4 e2 mei3 ling3 dao3 ren2 reng2 ran2 biao3 shi4 .",
	"ta1 men xi1 wang4 zai4 ai4 li4 she4 gong1 de gong1 zuo4 wu3 can1 shang4 hui4 mian4 .",
	"chu1 bu4 zuo4 wei4 biao3 xian3 shi4 .",
	"te4 lang3 pu3 bei4 an1 pai2 zai4 pu3 jing1 pang2 bian1 .",
	"dan4 zai4 sui2 hou4 jin4 xing2 de gong1 zuo4 wu3 can1 qi1 jian1 .",
	"zuo4 wei4 an1 pai2 xian3 ran2 yi3 jing1 fa1 sheng1 le bian4 hua4 .",
	"cong2 zhao4 pian1 lai2 kan4 .",
	"pu3 jing1 dang1 shi2 zheng4 quan2 shen2 guan4 zhu4 de yu3 lian2 he2 guo2 mi4 shu1 chang2 gu3 te4 lei2 si1 jiao1 tan2 .",
	"ou1 meng2 wei3 yuan2 hui4 zhu3 xi2 rong2 ke4 zuo4 zai4 pu3 jing1 de you4 bian1 .",
	"er2 te4 lang3 pu3 ze2 zuo4 zai4 ma3 ke4 long2 pang2 bian1 .",
	"ma3 ke4 long2 de you4 bian1 ze2 shi4 de2 guo2 zong3 li3 mo4 ke4 er3 .",

	"ci3 qian2 .",
	"pu3 jing1 zai4 fang3 wen4 ba1 li2 qi1 jian1 biao3 shi4 .",
	"ta1 bu4 pai2 chu2 ,",
	"yu3 te4 lang3 pu3 zai4 gong1 zuo4 wu3 can1 shi2 ,",
	"jin4 xing2 jiao1 liu2 .",
	"pu3 jing1 zai4 fa3 guo2 pin2 dao4 de jie2 mu4 zhong1 hui2 da2 ,",
	"shi4 fou3 yi3 tong2 te4 lang3 pu3 jin4 xing2 jiao1 liu2 de wen4 ti2 shi2 ,",
	"biao3 shi4 zan4 shi2 mei2 you3 .",
	"wo3 men zhi3 da3 le ge4 zhao1 hu1 .",
	"yi2 shi4 yi3 zhe4 yang4 de fang1 shi4 jin4 xing2 .",
	"wo3 men wu2 fa3 zai4 na4 li3 jin4 xing2 jiao1 liu2 .",
	"wo3 men guan1 kan4 le fa1 sheng1 de shi4 qing2 .",
	"dan4 xian4 zai4 hui4 you3 gong1 zuo4 wu3 can1 .",
	"ye3 xu3 zai4 na4 li3 .",
	"wo3 men hui4 jin4 xing2 jie1 chu4 .",
	"dan4 shi4 .",
	"wu2 lun4 ru2 he2 .",
	"wo3 men shang1 ding4 .",
	"wo3 men zai4 zhe4 li3 ,",
	"bu4 hui4 wei2 fan3 zhu3 ban4 guo2 de gong1 zuo4 an1 pai2 .",
	"gen1 ju4 ta1 men de yao1 qiu2 .",
	"wo3 men bu4 hui4 ,"
	"zai4 zhe4 li3 zu3 zhi1 ren4 he2 hui4 mian4 .",
	"er2 shi4 ke3 neng2 hui4 zai4 feng1 hui4 qi1 jian1 ,",
	"huo4 zai4 ci3 zhi1 hou4 ju3 xing2 hui4 mian4 .",
	"pu3 jing1 hai2 biao3 shi4 .",
	"e2 luo2 si1 zhun3 bei4 tong2 mei3 guo2 jin4 xing2 dui4 hua4 .",
	"fan3 zheng4 bu2 shi4 mo4 si1 ke1 yao4 tui4 chu1 zhong1 dao3 tiao2 yue1 .",

	# "bai2 jia1 xuan1 hou4 lai2 yin3 yi3 hao2 zhuang4 de shi4 yi1 sheng1 li3 qu3 guo4 qi1 fang2 nv3 ren2 . qu3 tou2 fang2 xi2 fu4 shi2 ta1 gang1 gang1 guo4 shi2 liu4 sui4 sheng1 ri4 .",
	# "na4 shi4 xi1 yuan2 shang4 gong3 jia1 cun1 da4 hu4 gong3 zeng1 rong2 de tou2 sheng1 nv3 , bi3 ta1 da4 liang3 sui4 .",
	# "ta1 zai4 wan2 quan2 wu2 zhi1 huang1 luan4 zhong1 du4 guo4 le xin1 hun1 zhi1 ye4 , liu2 xia4 le yong3 yuan3 xiu1 yu2 xiang4 ren2 dao4 ji2 de ke3 xiao4 de sha3 yang4 , er2 zi4 ji3 que4 yong3 sheng1 nan2 yi3 wang4 ji4 .",
	# "yi1 nian2 hou4 , zhe4 ge4 nv3 ren2 si3 yu2 nan2 chan3 . di4 er4 fang2 qu3 de shi4 nan2 yuan2 pang2 jia1 cun1 yin1 shi2 ren2 jia1 pang2 xiu1 rui4 de nai3 gan4 nv3 er2 .",
	# "zhe4 nv3 zi3 you4 zheng4 hao3 bi3 ta1 xiao3 liang3 sui4 , mu2 yang4 jun4 xiu4 yan3 jing1 hu1 ling2 er2 . ta1 wan2 quan2 bu4 zhi1 dao4 jia4 ren2 shi4 zen3 me hui2 shi4 , er2 ta1 ci3 shi2 yi3 an1 shu2 nan2 nv3 zhi1 jian1 suo3 you3 de yin3 mi4 .",
	# "ta1 kan4 zhe ta1 de xiu1 qie4 huang1 luan4 er2 xiang3 dao4 zi4 ji3 di4 yi1 ci4 de sha3 yang4 fan3 dao4 jue2 de geng4 fu4 ci4 ji1 .",
	# "dang1 ta1 hong1 suo1 zhe ba3 duo3 duo3 shan3 shan3 er2 you4 bu4 gan3 wei2 ao4 ta1 de xiao3 xi2 fu4 guo3 ru4 shen1 xia4 de shi2 hou4 , ta1 ting1 dao4 le ta1 de bu2 shi4 huan1 le4 er2 shi4 tong4 ku3 de yi1 sheng1 ku1 jiao4 .",
	# "dang1 ta1 pi2 bei4 de xie1 xi1 xia4 lai2 , cai2 fa1 jue2 jian1 bang3 nei4 ce4 teng2 tong4 zuan1 xin1 , ta1 ba3 ta1 yao3 lan4 le .",
	# "ta1 fu3 shang1 xi1 tong4 de shi2 hou4 , xin1 li3 jiu4 chao2 qi3 le dui4 zhe4 ge4 jiao1 guan4 de2 you3 dian3 ren4 xing4 de nai3 gan4 nv3 er2 de nao3 huo3 .",
	# "zheng4 yu4 fa1 zuo4 , ta1 que4 ban1 guo4 ta1 de jian1 bang3 an4 shi4 ta1 zai4 lai2 yi1 ci4 . yi1 dang1 jing1 guo4 nan2 nv3 jian1 de di4 yi1 ci4 jiao1 huan1 , ta1 jiu4 bian4 de2 mei2 you3 jie2 zhi4 de ren4 xing4 .",
	# "zhe4 ge4 nv3 ren2 cong2 xia4 jiao4 ding3 zhe hong2 chou2 gai4 jin1 jin4 ru4 bai2 jia1 men2 lou2 dao4 tang3 jin4 yi1 ju4 bao2 ban3 guan1 cai2 tai2 chu1 zhe4 ge4 men2 lou2 , shi2 jian1 shang4 bu4 zu2 yi1 nian2 , shi4 hai4 lao2 bing4 si3 de .",

	# "quan2 xin1 ao4 di2 A ba1 L",
	# "shi2 wan4 ju4 hui4",
	# "shi2 wan4 hao2 li3",
	# "hu1 shi4 fu4 li4 hua2 ting2 er4 qi1",
	# "shuang1 hu2 jing3 guan1",
	# "yao1 nin2 pin3 jian4",
	# "wu4 lv4 kong1 jian1 dong4 xue2 yan2 liao2",
	# "wei4 jian4 kang1 hu1 xi1 bao3 jia4 hu4 hang2",
	# "chong2 shang4 ke1 xue2 chuang4 wen2 ming2",
	# "fan3 dui4 xie2 jiao4 qi2 nu3 li4",
	# "zhong1 guo2 nong2 ye4 yin2 hang2",
	# "zhang3 shang4 yin2 hang2 APP mian3 fei4 chou1 xi2 dian3",
	# "qing2 shang1 gao1",
	# "cai2 neng2 tong4 kuai4 zuo4 zi4 ji3",
	# "ren2 sheng1 jiu4 xiang4 shi4 yi4 chang3 lv3 xing2",
	# "bu2 bi4 zai4 hu1 mu4 di4 di4",
	# "zai4 hu1 de5",
	# "shi4 yan2 tu2 de5 feng1 jing3",
	# "yi3 ji2 kan4 feng1 jing3 de5 xin1 qing2",
	# "wo4 er3 wo4 ban4 nin2 yi2 lu4 qian2 xing2",
	# "zhen1 zheng4 xi3 huan1 ni3 de5 ren2",
	# "er4 shi2 si4 xiao3 shi2 dou1 you3 kong4",
	# "xiang3 song4 ni3 de5 ren2",
	# "dong1 nan2 xi1 bei3 dou1 shun4 lu4",
	# "di1 di1 shun4 feng1 che1",
	# "wo3 men kan4 dao4 tai4 yang2 fa1 chu1 de5 guang1",
	# "xu1 yao4 ba1 fen1 zhong1",
	# "kan4 dao4 hai3 wang2 xing1 zhe2 she4 chu1 de5 guang1",
	# "xu1 yao4 si4 xiao3 shi2",
	# "kan4 dao4 yin2 he2 xi4 bian1 yuan2 de5 guang1",
	# "xu1 yao4 qi1 dian3 liu4 wan4 nian2",
	# "kan4 dao4 yu3 zhou4 zhong1",
	# "li2 wo3 men5 zui4 yuan3 de5 na4 ke1 xing1 xing5 fa1 chu1 de5 guang1",
	# "xu1 yao4 yi4 bai3 san1 shi2 jiu3 yi4 nian2",
	# "suo3 you3 de5 guang1 mang2",
	# "dou1 xu1 yao4 shi2 jian1 cai2 neng2 bei4 kan4 dao4",
	# "chui2 zi ye3 yi2 yang4",
	]

	)

def hparams_debug_string():
	values = hparams.values()
	hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
	return 'Hyperparameters:\n' + '\n'.join(hp)
