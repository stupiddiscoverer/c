#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define ROWS 9
#define COLS 9
#define MINES 10
char board[ROWS][COLS];    // 游戏显示的板
char revealed[ROWS][COLS]; // 已揭示的格子
int mines[ROWS][COLS];     // 记录地雷位置的数组
// 函数声明
void initializeBoard();
void placeMines();
void calculateNumbers();
void printBoard(int reveal);
void revealCell(int row, int col);
void gameLoop();

int main()
{
    srand(time(NULL)); // 用当前时间作为随机数种子
    initializeBoard();
    placeMines();
    calculateNumbers();
    gameLoop();
    return 0;
}

// 初始化棋盘
void initializeBoard()
{
    for (int i = 0; i < ROWS; i++)
    {
        for (int j = 0; j < COLS; j++)
        {
            board[i][j] = '.';
            revealed[i][j] = 0; // 0表示未揭示
            mines[i][j] = 0;    // 0表示没有地雷
        }
    }
}

// 随机放置地雷
void placeMines()
{
    int count = 0;
    while (count < MINES)
    {
        int r = rand() % ROWS;
        int c = rand() % COLS;
        if (mines[r][c] == 0)
        {
            mines[r][c] = 1; // 1表示当前位置有地雷
            count++;
        }
    }
}

// 计算每个格子周围地雷的数量
void calculateNumbers()
{
    for (int r = 0; r < ROWS; r++)
    {
        for (int c = 0; c < COLS; c++)
        {
            if (mines[r][c] == 1)
                continue;
            int count = 0;
            for (int dr = -1; dr <= 1; dr++)
            {
                for (int dc = -1; dc <= 1; dc++)
                {
                    if (r + dr >= 0 && r + dr < ROWS && c + dc >= 0 && c + dc < COLS)
                    {
                        count += mines[r + dr][c + dc];
                    }
                }
            }
            if (count > 0)
            {
                board[r][c] = '0' + count; // 将数字转为字符
            }
        }
    }
}

// 打印棋盘
void printBoard(int reveal)
{
    printf("  | ");
    for (int j = 0; j < COLS; j++)
    {
        printf("%d ", j);
    }
    printf("\n");
    for (int i = 0; i < ROWS; i++)
    {
        printf("%d | ", i);
        for (int j = 0; j < COLS; j++)
        {
            if (reveal)
            {
                if (mines[i][j] == 1)
                {
                    printf("* ");
                }
                else
                {
                    printf("%c ", board[i][j]);
                }
            }
            else
            {
                printf("%c ", revealed[i][j] ? board[i][j] : '.');
            }
        }
        printf("\n");
    }
}

// 揭示格子
void revealCell(int row, int col)
{
    if (row < 0 || row >= ROWS || col < 0 || col >= COLS)
        return; // 越界
    if (revealed[row][col])
        return;             // 已经揭示过
    revealed[row][col] = 1; // 标记为已揭示
    if (board[row][col] == '0')
    {
        // 如果是0，则揭示周围的格子
        for (int dr = -1; dr <= 1; dr++)
        {
            for (int dc = -1; dc <= 1; dc++)
            {
                if (!(dr == 0 && dc == 0))
                {
                    revealCell(row + dr, col + dc);
                }
            }
        }
    }
}

// 游戏循环
void gameLoop()
{
    int gameOver = 0;
    while (!gameOver)
    {
        printBoard(0);
        int row, col;
        printf("输入要揭示的格子 (格式: row col): ");
        scanf("%d %d", &row, &col);

        if (row < 0 || row >= ROWS || col < 0 || col >= COLS)
        {
            printf("无效的输入，请重新输入\n");
            continue;
        }

        if (mines[row][col] == 1)
        {
            printf("踩到地雷了！游戏结束！\n");
            gameOver = 1;
            printBoard(1);
        }
        else
        {
            revealCell(row, col);
        }

        // 检查是否胜利
        int revealedCount = 0;
        for (int i = 0; i < ROWS; i++)
        {
            for (int j = 0; j < COLS; j++)
            {
                if (revealed[i][j])
                    revealedCount++;
            }
        }
        if (revealedCount == ROWS * COLS - MINES)
        {
            printf("恭喜你赢得了游戏！\n");
            gameOver = 1;
            printBoard(1);
        }
    }

}