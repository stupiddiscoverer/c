//二叉树中的最大路径和

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
};

int maxVal;

int max(int a, int b, int c, int d)
{
    int max = a;
    if (max < b)
    {
        max = b;
    }
    if (max < c)
    {
        max = c;
    }
    if (max < d)
    {
        max = d;
    }
    return max;
}

int getMaxPathVal(struct TreeNode *node)
{
    int left, right;
    if (node->left == 0)
    {
        left = 0;
    }
    else
    {
        left = getMaxPathVal(node->left);
    }
    if (node->right == 0)
    {
        right = 0;
    }
    else
    {
        right = getMaxPathVal(node->right);
    }
    int tempMax = max(left+node->val, right+node->val, left+right+node->val, node->val);
    if (maxVal < tempMax)
    {
        maxVal = tempMax;
    }
    return max(left + node->val, right + node->val, node->val, node->val);
}

//每个节点的左右子树存在一个最大子路径，连接起来就是该节点的最大路径，找到整个树路径最大的节点就行
//第一步，计算每个节点的最大路径，第二步，找最大值返回
int maxPathSum(struct TreeNode* root) {
    //因为最少包含一个节点，如果所有节点都是负数，那也只能返回一个最大的负数了
    maxVal = root->val;
    getMaxPathVal(root);
    return maxVal;
}

int main(int argc, char const *argv[])
{
    struct TreeNode a = {0};
    struct TreeNode b = {0};
    struct TreeNode c = {0};

    a.val = 2;
    b.val = -1;
    c.val = -2;
    a.left = &b;
    a.right = &c;
    printf("%d\n", maxPathSum(&a));
    return 0;
}
