import random
import torch

def prob_subset_selection(arr, target_sum=[300,360], sample_k=4, max_attempts=50, default_subset=[1,3,5,8,13]):
    values = []
    indices = []
    attempts = 0

    while attempts < max_attempts:
            # 随机选择 k 个索引
        indices = random.sample(range(len(arr)), sample_k)
        # 获取对应的值
        values = [arr[i] for i in indices]
        total_val=sum([value**2 for value in values])
        if target_sum[0] <= total_val <= target_sum[-1]:
            break
        attempts += 1

    if len(values) > 0:
        # 对 subset 和 indices 进行排序
        values, indices = zip(*sorted(zip(values, indices)))
        values=list(values);indices=list(indices)
    else:
        values = default_subset
        indices = [arr.index(num) for num in default_subset]
    return values,indices


if __name__ == '__main__':
    # 示例使用
    arr = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)#68
    target_sum = [200,360]

    # 初始化一个字典来统计每个元素被选择的次数
    selection_counts = {num: 0 for num in arr}

    # 运行100次程序，统计每个元素被选择的次数
    num_iterations = 10000
    sum_subsets=[]
    for _ in range(num_iterations):
        subset,indexes = prob_subset_selection(arr, target_sum)
        print(subset,sum(subset),indexes)
        sum_subsets.append(sum(subset))
        for num in subset:
            selection_counts[num] += 1

    # 打印统计结果
    for num, count in selection_counts.items():
        print(f"Element {num} was selected {count} times.")

    print(max(sum_subsets),min(sum_subsets))

    # 打印每个元素被选择的概率
    print("\nSelection probabilities:")
    for num, count in selection_counts.items():
        print(f"Element {num} selection probability: {count / num_iterations:.2f}")
