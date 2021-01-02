import logger
import DQN
import logging
def main():
    logger.console_out("output.txt")

    logging.fatal("Next will output debug test/ info test/ warning test.")
    logging.fatal("Please check the settings of logging.")
    logging.debug("debug test") 
    logging.info("info test") 
    logging.warning("warning test") 

    hyper_param = {
        "seed": 42,  # which seed to use
        "env": "BreakoutNoFrameskip-v4",  # name of the game
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(1e6),# total number of steps to run the environment for
        "batch-size": 32,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 1,  # number of iterations between every optimization step
        "print-freq": 10,
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "backup_record_step": 10000, # frequence of saving network
        "use-prioritybuffer": True,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta0": 0.4,
        "prioritized_replay_eps": 1e-6,
        "use_multi_step": True,
        "use_noisy_network": True
    }

    player = DQN.DQNplayer(networkName="conv2d",hyper_param = hyper_param) #声明player，使用Conv2D解析RGB
    #player = DQN.DQNplayer(name="Breakout-v0",networkName="duel",max_memory_length = 20000)
    player.main_process() # 进行训练，训练过程中要绘制episode总奖励曲线
    # 选择保存模型
    player.savemodel("Q_main")
    # 能够跑出一个模型

if __name__ == "__main__":
    main()