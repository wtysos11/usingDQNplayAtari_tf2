import logger
import DQN
import tensorflow as tf

def main():
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)
    logger.console_out("output.txt")

    #player = DQN.DQNplayer(name="Pong-v0",networkName="conv2d") #声明player，使用Conv2D解析RGB
    player = DQN.DQNplayer(name="Breakout-v0",networkName="conv2d")
    player.main_process() # 进行训练，训练过程中要绘制episode总奖励曲线
    # 选择保存模型
    player.savemodel("Q_main")
    # 能够跑出一个模型

if __name__ == "__main__":
    main()