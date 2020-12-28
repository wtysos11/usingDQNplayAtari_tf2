import logger
import DQN

def main():
    logger.console_out("output.txt")

    player = DQN.DQNplayer(name="Pong-v0",networkName="conv2d") #声明player，使用Conv2D解析RGB
    player.main_process() # 进行训练，训练过程中要绘制episode总奖励曲线
    # 选择保存模型
    # 能够跑出一个模型

if __name__ == "__main__":
    main()