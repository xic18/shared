dataframe高效操作



我有一个datafram，有一列是key，这个key有100000种不同的key，每个key都对应10行数据，有没有办法在O(1)的复杂度下，给定某一个key，把这10行数据拿出来？



当然可以！要实现 **O(1) 时间复杂度** 的查找，你需要预先建立一个索引字典。以下是几种高效的方法：

## 方法1：使用groupby()预先构建索引字典（推荐）

这是最直接有效的方法，只需要一次性的O(n)预处理，之后每次查询都是O(1)。

python

```
import pandas as pd
import numpy as np

# 示例数据生成（1,000,000行，100,000个key，每个key对应10行）
np.random.seed(42)
n_keys = 100000
data_size = 1000000

# 创建测试DataFrame
keys = np.repeat(np.arange(n_keys), 10)  # 每个key重复10次
values = np.random.randn(data_size)

df = pd.DataFrame({
    'key': keys,
    'value': values
})

# 预先构建索引字典 - O(n) 一次性操作
index_dict = {}
for key, group in df.groupby('key'):
    index_dict[key] = group.index  # 存储索引位置

# O(1) 查询函数
def get_rows_by_key_fast(key):
    """O(1)复杂度获取指定key的所有行"""
    if key in index_dict:
        return df.loc[index_dict[key]]
    else:
        return pd.DataFrame()

# 使用示例
result = get_rows_by_key_fast(42)
print(f"Key 42 对应的 {len(result)} 行数据:")
print(result.head())
```



## 方法2：使用字典缓存DataFrame切片

如果你内存充足，可以直接缓存整个数据切片：

python

```
# 预先缓存所有数据切片 - 占用更多内存，但速度最快
cache_dict = {}
for key, group in df.groupby('key'):
    cache_dict[key] = group  # 直接缓存DataFrame切片

# O(1) 查询
def get_rows_by_key_instant(key):
    """最快的O(1)查询，直接返回缓存的DataFrame"""
    return cache_dict.get(key, pd.DataFrame())

# 使用示例
result = get_rows_by_key_instant(42)
```



## 方法3：使用MultiIndex（如果你需要频繁按key查询）

将key设为索引的一部分，可以加速查询：

python

```
# 设置MultiIndex
df_indexed = df.set_index('key', append=True).swaplevel(0, 1).sort_index()

# 查询速度会很快（虽然不是严格的O(1)，但非常高效）
result = df_indexed.loc[42]
```



## 性能对比

| 方法                | 预处理时间 | 单次查询时间 | 内存占用 | 适用场景               |
| ------------------- | ---------- | ------------ | -------- | ---------------------- |
| 方法1（索引字典）   | O(n)       | O(1)         | 中等     | 通用推荐               |
| 方法2（完整缓存）   | O(n)       | O(1)         | 较高     | 内存充足，要求极致速度 |
| 方法3（MultiIndex） | O(n log n) | O(log n)     | 较低     | 需要多种查询方式       |

## 推荐方案

**对于你的需求（100,000个key，每个key10行数据），我强烈推荐方法1：**

python

```
class KeyIndexedDataFrame:
    def __init__(self, df, key_column='key'):
        self.df = df
        self.key_column = key_column
        self._build_index()
    
    def _build_index(self):
        """构建索引字典"""
        self.index_dict = {}
        for key, group in self.df.groupby(self.key_column):
            self.index_dict[key] = group.index
    
    def get_rows(self, key):
        """O(1)复杂度获取指定key的所有行"""
        if key in self.index_dict:
            return self.df.loc[self.index_dict[key]]
        return pd.DataFrame()
    
    def get_values(self, key, column):
        """O(1)复杂度获取指定key的某列值"""
        if key in self.index_dict:
            return self.df.loc[self.index_dict[key], column]
        return pd.Series()

# 使用示例
indexed_df = KeyIndexedDataFrame(df, 'key')
result = indexed_df.get_rows(42)
```



这种方法只需要一次性的预处理，之后所有查询都是真正的O(1)时间复杂度！