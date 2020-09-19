import fqt_dqn_agent
import matplotlib.pyplot as plt
import sys
import os

def main():
    agent = fqt_dqn_agent.FQT_DQN_Agent()
    #* Start Reinforced Learning
    #rl_visualized(agent, 'end-to-end_trial')

    #* Watch model
    agent.load_model('__end-to-end_trial/current_trained_dqn_model.npy')
    agent.watch(60)

def rl_visualized(agent, session_name):
    #? Graph visualization
    scores_list = []
    epsilons_list = []

    plt.style.use('dark_background') # Dark mode
    plt.title("Scores over Training Episodes")
    plt.xlabel("Episodes")
    plt.ylabel('Scores')
    plt.plot([], scores_list, label="Current Score", color="lime")
    plt.plot([], epsilons_list, label="Epsilon Value (%)", color="slateblue")
    plt.legend(loc='upper left', prop={'size':8})
    PLOT_GRAPH_EVERY = 1
    SAVE_GRAPH_EVERY = 25

    #* Reinforcement Learning - Agent Training of N Episodes
    SAVE_MODEL_EVERY = 25

    #! Session
    session_name = "__" + session_name
    create_session_folder(session_name)
    #ask_session(session_name)
    agent.load_model(session_name + "/" + 'current_trained_dqn_model.npy')

    print("[*] Started Fixed-Q-Target DQN Reinforcement Learning.")
    for episode in range(0,300):
        current_score = agent.learn()

        #? Storing info for plot
        epsilons_list.append(agent.epsilon*100) # % Representation
        scores_list.append(current_score)

        #if (episode % 50) == 0 and episode != 0:
        #    print("[>] 50 episodes have passed. Current average score: {AVG_SCORE}".format(AVG_SCORE=sum(scores_list)/(episode+1)))
        #    print("[*] Epsilon: {EPSILON}".format(EPSILON=agent.epsilon))

        #* Save trained model every N episode
        if (episode % SAVE_MODEL_EVERY) == 0 and episode != 0:
            agent.save_model(session_name + "/" + 'current_trained_dqn_model')

        # Plot progress
        if (episode % PLOT_GRAPH_EVERY) == 0 and episode != 0:
            plt.plot(list(range(0,episode+1)), scores_list, label="Current Score", color="lime")
            plt.plot(list(range(0,episode+1)), epsilons_list, label="Epsilon Value (%)", color="slateblue")
            plt.pause(0.001)

        # Save Graph
        if (episode % SAVE_GRAPH_EVERY) == 0 and episode != 0:
            plt.savefig(session_name + "/" + 'scores_over_episode.png')

def ask_session(session_name):
    yes_list = ['y', 'yes', 'yep']
    no_list = ['n', 'no', 'nope']

    while True:
        print("[?] Use existing session {SESSION_NAME} (Y/N)? -> ".format(SESSION_NAME=session_name))
        response = input()
       
        if response.lower() in yes_list:
            return
        elif response.lower() in no_list:
            print("[!] Since session is not to be used, please create a new one. Exiting program...")
            sys.exit(1)
        else:
            print("Invalid response.")

def create_session_folder(session_name):
    if not os.path.exists(session_name):
        print("[+] Directory not found. Creating new folder for session {SESSION_NAME}...".format(SESSION_NAME=session_name))
        os.makedirs(session_name)
    return

if __name__ == "__main__":
    main()