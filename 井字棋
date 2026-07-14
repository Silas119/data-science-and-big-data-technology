#include <iostream>
#include <vector>
#include <array>
#include <unordered_map>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>

using namespace std;

// 定义棋盘状态
const int EMPTY = 0;   // 空格
const int HUMAN = 1;   // 人类玩家，使用 X
const int AI = -1;     // AI玩家，使用 O

// 全局随机数生成器，用于随机落子、随机探索等
mt19937 rng(random_device{}());

/*
    TicTacToe结构体
    作用：
    1. 保存井字棋棋盘
    2. 判断合法动作
    3. 执行落子
    4. 判断胜负
    5. 将棋盘状态编码为字符串，供Q-Learning使用
*/
struct TicTacToe {
    // 使用一维数组表示3×3棋盘
    // 位置对应关系：
    // 0 1 2
    // 3 4 5
    // 6 7 8
    array<int, 9> board{};

    // 构造函数，创建对象时初始化棋盘
    TicTacToe() {
        reset();
    }

    // 重置棋盘为空
    void reset() {
        board.fill(EMPTY);
    }

    // 获取当前棋盘中所有可以落子的位置
    vector<int> legalActions() const {
        vector<int> actions;

        for (int i = 0; i < 9; i++) {
            if (board[i] == EMPTY) {
                actions.push_back(i);
            }
        }

        return actions;
    }

    // 执行落子
    // pos表示落子位置，player表示当前玩家
    bool makeMove(int pos, int player) {
        if (pos < 0 || pos >= 9 || board[pos] != EMPTY) {
            return false;
        }

        board[pos] = player;
        return true;
    }

    // 判断棋盘是否已经下满
    bool isFull() const {
        for (int cell : board) {
            if (cell == EMPTY) {
                return false;
            }
        }

        return true;
    }

    // 判断当前棋盘是否有获胜者
    int winner() const {
        // 所有可能的获胜连线
        static const int lines[8][3] = {
            {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
            {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
            {0, 4, 8}, {2, 4, 6}
        };

        for (auto& line : lines) {
            int a = line[0];
            int b = line[1];
            int c = line[2];

            if (board[a] != EMPTY &&
                board[a] == board[b] &&
                board[b] == board[c]) {
                return board[a];
            }
        }

        return EMPTY;
    }

    // 判断游戏是否结束
    bool gameOver() const {
        return winner() != EMPTY || isFull();
    }

    // 将棋盘状态编码为字符串
    // AI用O表示，HUMAN用X表示，空格用.表示
    string encode() const {
        string s;
        s.reserve(9);

        for (int cell : board) {
            if (cell == AI) {
                s.push_back('O');
            }
            else if (cell == HUMAN) {
                s.push_back('X');
            }
            else {
                s.push_back('.');
            }
        }

        return s;
    }

    // 打印棋盘
    void print() const {
        cout << "\n";

        for (int i = 0; i < 9; i++) {
            char ch;

            if (board[i] == HUMAN) {
                ch = 'X';
            }
            else if (board[i] == AI) {
                ch = 'O';
            }
            else {
                ch = char('1' + i);
            }

            cout << " " << ch << " ";

            if (i % 3 != 2) {
                cout << "|";
            }
            else if (i != 8) {
                cout << "\n---+---+---\n";
            }
        }

        cout << "\n\n";
    }
};

/*
    QLearningAgent结构体
    作用：
    1. 保存Q表
    2. 使用ε-greedy策略选择动作
    3. 根据奖励更新Q值
    4. 保存和读取Q表
*/
struct QLearningAgent {
    // Q表：状态字符串 -> 9个动作对应的Q值
    unordered_map<string, array<double, 9>> Q;

    double alpha = 0.1;          // 学习率
    double gamma = 0.9;          // 折扣因子
    double epsilon = 1.0;        // 探索概率
    double epsilonMin = 0.05;    // 最小探索概率
    double epsilonDecay = 0.99995; // 探索概率衰减系数

    // 获取某个状态对应的Q值数组
    array<double, 9>& getStateQ(const string& state) {
        if (Q.find(state) == Q.end()) {
            Q[state] = {};
            Q[state].fill(0.0);
        }

        return Q[state];
    }

    // 从合法动作中随机选择一个动作
    int randomChoice(const vector<int>& actions) {
        uniform_int_distribution<int> dist(0, (int)actions.size() - 1);
        return actions[dist(rng)];
    }

    // 根据ε-greedy策略选择动作
    int chooseAction(const string& state, const vector<int>& legalActions, bool training) {
        if (legalActions.empty()) {
            return -1;
        }

        // 训练阶段，以epsilon概率随机探索
        if (training) {
            uniform_real_distribution<double> prob(0.0, 1.0);

            if (prob(rng) < epsilon) {
                return randomChoice(legalActions);
            }
        }

        // 非探索情况下，选择Q值最大的动作
        auto& qValues = getStateQ(state);

        double bestValue = -numeric_limits<double>::infinity();
        vector<int> bestActions;

        for (int action : legalActions) {
            double value = qValues[action];

            if (value > bestValue) {
                bestValue = value;
                bestActions.clear();
                bestActions.push_back(action);
            }
            else if (value == bestValue) {
                bestActions.push_back(action);
            }
        }

        return randomChoice(bestActions);
    }

    // 根据Q-Learning公式更新Q值
    void update(
        const string& state,
        int action,
        double reward,
        const string& nextState,
        const vector<int>& nextLegalActions
    ) {
        auto& qValues = getStateQ(state);
        double oldValue = qValues[action];

        double nextMax = 0.0;

        if (!nextLegalActions.empty()) {
            auto& nextQValues = getStateQ(nextState);
            nextMax = -numeric_limits<double>::infinity();

            for (int nextAction : nextLegalActions) {
                nextMax = max(nextMax, nextQValues[nextAction]);
            }
        }

        // Q(s,a) = Q(s,a) + alpha * [reward + gamma * maxQ(s',a') - Q(s,a)]
        qValues[action] = oldValue + alpha * (reward + gamma * nextMax - oldValue);
    }

    // 衰减epsilon，使AI从多探索逐渐转向多利用
    void decayEpsilon() {
        epsilon = max(epsilonMin, epsilon * epsilonDecay);
    }

    // 保存Q表到文件
    void save(const string& filename) {
        ofstream out(filename);

        if (!out) {
            cerr << "无法保存 Q 表到文件: " << filename << "\n";
            return;
        }

        for (const auto& pair : Q) {
            out << pair.first;

            for (double v : pair.second) {
                out << " " << v;
            }

            out << "\n";
        }

        cout << "Q 表已保存到 " << filename << "\n";
    }

    // 从文件读取Q表
    void load(const string& filename) {
        ifstream in(filename);

        if (!in) {
            cerr << "无法读取 Q 表文件: " << filename << "\n";
            return;
        }

        Q.clear();

        string line;
        while (getline(in, line)) {
            stringstream ss(line);
            string state;
            ss >> state;

            array<double, 9> values{};
            values.fill(0.0);

            for (int i = 0; i < 9; i++) {
                ss >> values[i];
            }

            Q[state] = values;
        }

        cout << "Q 表已从 " << filename << " 读取，状态数量: " << Q.size() << "\n";
    }
};

// 随机玩家落子
int randomMove(const TicTacToe& game) {
    vector<int> actions = game.legalActions();
    uniform_int_distribution<int> dist(0, (int)actions.size() - 1);
    return actions[dist(rng)];
}

// Minimax评分函数
int minimaxScore(const TicTacToe& game, int depth) {
    int w = game.winner();

    if (w == HUMAN) {
        return 10 - depth;
    }

    if (w == AI) {
        return depth - 10;
    }

    if (game.isFull()) {
        return 0;
    }

    return 0;
}

// Minimax递归搜索
int minimax(TicTacToe game, int currentPlayer, int depth) {
    if (game.gameOver()) {
        return minimaxScore(game, depth);
    }

    vector<int> actions = game.legalActions();

    if (currentPlayer == HUMAN) {
        int bestScore = -numeric_limits<int>::max();

        for (int action : actions) {
            TicTacToe nextGame = game;
            nextGame.makeMove(action, HUMAN);

            int score = minimax(nextGame, AI, depth + 1);
            bestScore = max(bestScore, score);
        }

        return bestScore;
    }
    else {
        int bestScore = numeric_limits<int>::max();

        for (int action : actions) {
            TicTacToe nextGame = game;
            nextGame.makeMove(action, AI);

            int score = minimax(nextGame, HUMAN, depth + 1);
            bestScore = min(bestScore, score);
        }

        return bestScore;
    }
}

// Minimax玩家选择动作
int minimaxMove(const TicTacToe& game) {
    vector<int> actions = game.legalActions();

    int bestScore = -numeric_limits<int>::max();
    vector<int> bestActions;

    for (int action : actions) {
        TicTacToe nextGame = game;
        nextGame.makeMove(action, HUMAN);

        int score = minimax(nextGame, AI, 0);

        if (score > bestScore) {
            bestScore = score;
            bestActions.clear();
            bestActions.push_back(action);
        }
        else if (score == bestScore) {
            bestActions.push_back(action);
        }
    }

    uniform_int_distribution<int> dist(0, (int)bestActions.size() - 1);
    return bestActions[dist(rng)];
}

// 训练AI对战随机玩家
void trainAgainstRandom(QLearningAgent& agent, int episodes) {
    TicTacToe game;

    for (int episode = 1; episode <= episodes; episode++) {
        game.reset();

        uniform_int_distribution<int> firstDist(0, 1);
        int currentPlayer = firstDist(rng) == 0 ? AI : HUMAN;

        string lastState;
        int lastAction = -1;
        bool hasLastAIAction = false;

        while (!game.gameOver()) {
            if (currentPlayer == AI) {
                string state = game.encode();
                vector<int> actions = game.legalActions();

                int action = agent.chooseAction(state, actions, true);
                game.makeMove(action, AI);

                int w = game.winner();

                if (w == AI) {
                    agent.update(state, action, 1.0, game.encode(), {});
                    break;
                }

                if (game.isFull()) {
                    agent.update(state, action, 0.0, game.encode(), {});
                    break;
                }

                lastState = state;
                lastAction = action;
                hasLastAIAction = true;

                currentPlayer = HUMAN;
            }
            else {
                int action = randomMove(game);
                game.makeMove(action, HUMAN);

                int w = game.winner();

                if (w == HUMAN) {
                    if (hasLastAIAction) {
                        agent.update(lastState, lastAction, -1.0, game.encode(), {});
                    }
                    break;
                }

                if (game.isFull()) {
                    if (hasLastAIAction) {
                        agent.update(lastState, lastAction, 0.0, game.encode(), {});
                    }
                    break;
                }

                if (hasLastAIAction) {
                    string nextState = game.encode();
                    vector<int> nextActions = game.legalActions();
                    agent.update(lastState, lastAction, 0.0, nextState, nextActions);
                }

                currentPlayer = AI;
            }
        }

        agent.decayEpsilon();

        if (episode % 10000 == 0) {
            cout << "训练进度: " << episode
                << " / " << episodes
                << ", epsilon = " << agent.epsilon
                << ", Q状态数 = " << agent.Q.size()
                << "\n";
        }
    }
}

// 训练AI对战随机玩家和Minimax玩家
void trainAgainstMixed(QLearningAgent& agent, int episodes, double minimaxRatio) {
    TicTacToe game;

    for (int episode = 1; episode <= episodes; episode++) {
        game.reset();

        uniform_int_distribution<int> firstDist(0, 1);
        int currentPlayer = firstDist(rng) == 0 ? AI : HUMAN;

        uniform_real_distribution<double> ratioDist(0.0, 1.0);
        bool useMinimax = ratioDist(rng) < minimaxRatio;

        string lastState;
        int lastAction = -1;
        bool hasLastAIAction = false;

        while (!game.gameOver()) {
            if (currentPlayer == AI) {
                string state = game.encode();
                vector<int> actions = game.legalActions();

                int action = agent.chooseAction(state, actions, true);
                game.makeMove(action, AI);

                int w = game.winner();

                if (w == AI) {
                    agent.update(state, action, 1.0, game.encode(), {});
                    break;
                }

                if (game.isFull()) {
                    agent.update(state, action, 0.0, game.encode(), {});
                    break;
                }

                lastState = state;
                lastAction = action;
                hasLastAIAction = true;

                currentPlayer = HUMAN;
            }
            else {
                int action;

                if (useMinimax) {
                    action = minimaxMove(game);
                }
                else {
                    action = randomMove(game);
                }

                game.makeMove(action, HUMAN);

                int w = game.winner();

                if (w == HUMAN) {
                    if (hasLastAIAction) {
                        agent.update(lastState, lastAction, -1.0, game.encode(), {});
                    }
                    break;
                }

                if (game.isFull()) {
                    if (hasLastAIAction) {
                        agent.update(lastState, lastAction, 0.0, game.encode(), {});
                    }
                    break;
                }

                if (hasLastAIAction) {
                    string nextState = game.encode();
                    vector<int> nextActions = game.legalActions();
                    agent.update(lastState, lastAction, 0.0, nextState, nextActions);
                }

                currentPlayer = AI;
            }
        }

        agent.decayEpsilon();

        if (episode % 10000 == 0) {
            cout << "混合训练进度: " << episode
                << " / " << episodes
                << ", epsilon = " << agent.epsilon
                << ", Q状态数 = " << agent.Q.size()
                << ", Minimax比例 = " << minimaxRatio
                << "\n";
        }
    }
}

// 获取人类玩家输入
int getHumanMove(const TicTacToe& game) {
    while (true) {
        cout << "请输入你的落子位置 1-9: ";

        int pos;
        cin >> pos;

        if (!cin) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "输入无效，请输入 1 到 9 的数字。\n";
            continue;
        }

        pos -= 1;

        if (pos < 0 || pos >= 9) {
            cout << "位置必须在 1 到 9 之间。\n";
            continue;
        }

        if (game.board[pos] != EMPTY) {
            cout << "这个位置已经有棋子了，请重新选择。\n";
            continue;
        }

        return pos;
    }
}

// 人类玩家与AI对战
void playHumanVsAI(QLearningAgent& agent) {
    TicTacToe game;
    game.reset();

    cout << "你是 X，AI 是 O。\n";

    char choice;
    cout << "你是否先手？输入 y/n: ";
    cin >> choice;

    int currentPlayer = (choice == 'y' || choice == 'Y') ? HUMAN : AI;

    while (!game.gameOver()) {
        game.print();

        if (currentPlayer == HUMAN) {
            int move = getHumanMove(game);
            game.makeMove(move, HUMAN);
            currentPlayer = AI;
        }
        else {
            string state = game.encode();
            vector<int> actions = game.legalActions();

            int move = agent.chooseAction(state, actions, false);

            if (move == -1) {
                break;
            }

            cout << "AI 落子位置: " << move + 1 << "\n";
            game.makeMove(move, AI);
            currentPlayer = HUMAN;
        }
    }

    game.print();

    int w = game.winner();

    if (w == HUMAN) {
        cout << "你赢了！\n";
    }
    else if (w == AI) {
        cout << "AI 赢了！\n";
    }
    else {
        cout << "平局！\n";
    }
}

// 主函数，提供菜单操作
int main() {
    QLearningAgent agent;

    while (true) {
        cout << "1. 训练 AI：对随机玩家\n";
        cout << "2. 人类 vs AI\n";
        cout << "3. 保存 Q 表\n";
        cout << "4. 读取 Q 表\n";
        cout << "5. 训练 AI：随机 + Minimax 混合训练\n";
        cout << "6. 退出\n";

        int option;
        cin >> option;

        if (!cin) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "输入无效。\n";
            continue;
        }

        if (option == 1) {
            int episodes;
            cout << "请输入训练局数，例如 100000: ";
            cin >> episodes;

            trainAgainstRandom(agent, episodes);

            cout << "训练完成。\n";
            cout << "当前 Q 状态数: " << agent.Q.size() << "\n";
            cout << "当前 epsilon: " << agent.epsilon << "\n";
        }
        else if (option == 2) {
            if (agent.Q.empty()) {
                cout << "警告：Q 表为空，AI 还没有训练。\n";
                cout << "建议先选择 1 训练 AI，或选择 4 读取已有 Q 表。\n";
            }

            playHumanVsAI(agent);
        }
        else if (option == 3) {
            agent.save("q_table.txt");
        }
        else if (option == 4) {
            agent.load("q_table.txt");
        }
        else if (option == 5) {
            int episodes;
            double minimaxRatio;

            cout << "请输入训练局数，例如 100000: ";
            cin >> episodes;

            cout << "请输入 Minimax 对手比例，例如 0.3 表示 30% 局数对 Minimax: ";
            cin >> minimaxRatio;

            if (minimaxRatio < 0.0) {
                minimaxRatio = 0.0;
            }

            if (minimaxRatio > 1.0) {
                minimaxRatio = 1.0;
            }

            trainAgainstMixed(agent, episodes, minimaxRatio);

            cout << "混合训练完成。\n";
            cout << "当前 Q 状态数: " << agent.Q.size() << "\n";
            cout << "当前 epsilon: " << agent.epsilon << "\n";
        }
        else if (option == 6) {
            cout << "退出程序。\n";
            break;
        }
        else {
            cout << "无效选项。\n";
        }
    }

    return 0;
}
