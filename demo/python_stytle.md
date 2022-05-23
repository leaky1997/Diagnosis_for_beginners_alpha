[TOC]
# 1.背景
为了帮助大家构建美观而统一的代码，请先参考本文档进行初步学习，本文档参考[谷歌python风格](https://github.com/shendeguize/GooglePythonStyleGuideCN)进行一定的节选。

__注意__ 本文档不是python教程，只是代码规范参考。 
# 2.Python 部分建议
## 2.1 import
* ` import x`（当`x`是包或模块）
* `from x import y` （当`x`是包前缀，`y`是不带前缀的模块名）
* `from x import  y as z` （当有重复模块名`y`或`y`过长不利于引用的时候）
* `import y as z` （仅在非常通用的简写的时候使用例如`import numpy as np`）
* 尽量不要使用相对引用
## 2.2 包的import
所有的新代码都要从完整包名来import模块

import示例应该像这样:

**Yes:**

```Python
# Reference absl.flags in code with the complete name (verbose).
# 在代码中使用完整路径调用absl.flags
import absl.flagsfrom doctor.who import jodie

FLAGS = absl.flags.FLAGS
```
``` Python
# Reference flags in code with just the module name (common).
# 在代码中只用包名来调用flags
from absl import flagsfrom doctor.who import jodie

FLAGS = flags.FLAGS
```

**No，不要使用相对路径:**(假设文件在`doctor/who`中,`jodie.py`也在这里)

```Python
# Unclear what module the author wanted and what will be imported.  The actual
# import behavior depends on external factors controlling sys.path.
# Which possible jodie module did the author intend to import?
# 不清楚作者想要哪个包以及最终import的是哪个包,
# 实际的import操作依赖于受到外部参数控制的sys.path
# 那么哪一个可能的jodie模块是作者希望import的呢?
import jodie
```
## 2.3 全局变量
尽量避免全局变量，如需使用，参考下面：

作为技术变量,模块级别的常量是允许并鼓励使用的.例如`MAX_HOLY_HANDGRENADE_COUNT = 3`, 常量必须由大写字母和下划线组成,参见下方[命名规则](https://google.github.io/styleguide/pyguide.html#s3.16-naming)

如果需要,全局变量需要在模块级别声明,并且通过在变量名前加`_`来使其对模块内私有化.外部对模块全局变量的访问必须通过公共模块级别函数,参见下方[命名规则](https://google.github.io/styleguide/pyguide.html#s3.16-naming)


## 2.4 列表推导和生成器表达式在简单情况下可用

```Python
result = [mapping_expr for value in iterable if filter_expr]

result = [{'key': value} for value in iterable
          if a_long_filter_expression(value)]

result = [complicated_transform(x)
          for x in iterable if predicate(x)]

descriptive_name = [
    transform({'key': key, 'value': value}, color='black')
    for key, value in generate_iterable(some_input)
    if complicated_condition_is_met(key, value)
]

result = []
for x in range(10):
    for y in range(5):
        if x * y > 10:
            result.append((x, y))

return {x: complicated_transform(x)
        for x in long_generator_function(parameter)
        if x is not None}

squares_generator = (x**2 for x in range(10))

unique_names = {user.name for user in users if user is not None}

eat(jelly_bean for jelly_bean in jelly_beans
    if jelly_bean.color == 'black')
```

**No:**

```Python
result = [complicated_transform(
          x, some_argument=x+1)
          for x in iterable if predicate(x)]

result = [(x, y) for x in range(10) for y in range(5) if x * y > 10]

return ((x, y, z)
        for x in range(5)
        for y in range(5)
        if x != y
        for z in range(5)
        if y != z)
```

## 2.5 其他注意事项
- 函数中的类、函数使允许的
- 可以使用简单的迭代器和运算符
- 生成器`yield`
- lambda 表达式
- 条件表达式(也称为三元运算符)是一种更短替代if语句的机制.例如`x = 1 if cond else 2`
- 建议每个函数的参数定义默认值例如`def foo(a=1, b=0):`
- 比较进阶的[装饰器](https://www.runoob.com/w3cnote/python-func-decorators.html)
- 在布尔环境下,Python对某些值判定为False,一个快速的经验规律是所有"空"值都被认为是False,所以`0, None, [], {}, ''`的布尔值都是False
- 尽可能利用字符串方法而非`string`模块.使用函数调用语法而非`apply`.在函数参数本就是一个行内匿名函数的时候,使用列表推导表达式和for循环而非`filter`和`map`
- 尽量避免使用过于强大的特性，过于强大意味着难以理解
  




# 3.代码风格规范
## 3.1 注释和文档字符串
确保使用正确的模块,函数,方法的文档字符串和行内注释.


#### 3.1.1 文档字符串
Python使用*文档字符串*来为代码生成文档.文档字符串是包,模块,类或函数的首个语句.这些字符串能够自动被`__doc__`成员方法提取并且被`pydoc`使用.(尝试在你的模块上运行`pydoc`来看看具体是什么).文档字符串使用三重双引号`"""`(根据[PEP-257](https://www.google.com/url?sa=D&q=http://www.python.org/dev/peps/pep-0257/)).文档字符串应该这样组织:一行总结(或整个文档字符串只有一行)并以句号,问好或感叹号结尾.随后是一行空行,随后是文档字符串,并与第一行的首个引号位置相对齐.更多具体格式规范如下.

#### 3.1.2 模块
每个文件都应包含许可模板.选择合适的许可模板用于项目(例如
Apache 2.0,BSD,LGPL,GPL)

文档应该以文档字符串开头,并描述模块的内容和使用方法.

```Python
"""A one line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

  Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
```

#### 3.1.3 函数和方法
在本节,"函数"所指包括方法,函数或者生成器.

函数应有文档字符串,除非符合以下所有条件:
* 外部不可见
* 非常短
* 简明

文档字符串应该包含足够的信息以在无需阅读函数代码的情况下调用函数.文档字符串应该是叙事体(`"""Fetches rows from a Bigtable."""`)的而非命令式的(`"""Fetch rows from a Bigtable."""`),除了`@property`(应与[attribute](https://google.github.io/styleguide/pyguide.html#384-classes)使用同样的风格).文档字符串应描述函数的调用语法和其意义,而非实现.对比较有技巧的地方,在代码中使用注释更合适.

覆写了基类的方法可有简单的文档字符串向读者指示被覆写方法的文档字符串例如`"""See base class."""`.这是因为没必要在很多地方重复已经在基类的文档字符串中存在的文档.不过如果覆写的方法行为实际上与被覆写方法不一致,或者需要提供细节(例如文档中表明额外的副作用),覆写方法的文档字符串至少要提供这些差别.

一个函数的不同方面应该在特定对应的分节里写入文档,这些分节如下.每一节都由以冒号结尾的一行开始, 每一节除了首行外,都应该以2或4个空格缩进并在整个文档内保持一致(译者建议4个空格以维持整体一致).如果函数名和签名足够给出足够信息并且能够刚好被一行文档字符串所描述,那么可以忽略这些节.

[*Args:*](https://google.github.io/styleguide/pyguide.html#doc-function-args)

列出每个参数的名字.名字后应有为冒号和空格,后跟描述.如果描述太长不能够在80字符的单行内完成.那么分行并缩进2或4个空格且与全文档一致(译者同样建议4个空格)

描述应该包含参数所要求的类型,如果代码不包含类型注释的话.如果函数容许`*foo`(不定长度参数列表)或`**bar`(任意关键字参数).那么就应该在文档字符串中列举为`*foo`和`**bar`.
    
[*Returns:(或对于生成器是Yields:)*](https://google.github.io/styleguide/pyguide.html#doc-function-returns)

描述返回值的类型和含义.如果函数至少返回None,这一小节不需要.如果文档字符串以Returns或者Yields开头(例如`"""Returns row from Bigtable as a tuple of strings."""`)或首句足够描述返回值的情况下这一节可忽略.
    
[*Raises:*](https://google.github.io/styleguide/pyguide.html#doc-function-returns)

列出所有和接口相关的异常.对于违反文档要求而抛出的异常不应列出.(因为这会矛盾地使得违反接口要求的行为成为接口的一部分)

```Python
def fetch_bigtable_rows(big_table, keys, other_silly_variable=None):
    """Fetches rows from a Bigtable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by big_table.  Silly things may happen if
    other_silly_variable is not None.

    Args:
        big_table: An open Bigtable Table instance.
        keys: A sequence of strings representing the key of each table row
            to fetch.
        other_silly_variable: Another optional variable, that has a much
            longer name than the other args, and which does nothing.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {'Serak': ('Rigel VII', 'Preparer'),
         'Zim': ('Irk', 'Invader'),
         'Lrrr': ('Omicron Persei 8', 'Emperor')}

        If a key from the keys argument is missing from the dictionary,
        then that row was not found in the table.

    Raises:
        IOError: An error occurred accessing the bigtable.Table object.
    """
```

#### 3.1.4 类
类定义下一行应为描述这个类的文档字符串.如果类有公共属性,应该在文档字符串中的`Attributes`节中注明,并且和[函数的`Args`](https://google.github.io/styleguide/pyguide.html#doc-function-args)一节风格统一.

```Python
class SampleClass(object):
    """Summary of class here.

    Longer class information....
    Longer class information....

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self, likes_spam=False):
        """Inits SampleClass with blah."""
        self.likes_spam = likes_spam
        self.eggs = 0

    def public_method(self):
        """Performs operation blah."""
```

#### 3.1.5 块注释和行注释
最后要在代码中注释的地方是代码技巧性的部分.如果你将要在下次[code review](http://en.wikipedia.org/wiki/Code_review)中揭示代码.应该现在就添加注释.在复杂操作开始前,注释几行.对于不够明晰的代码在行尾注释.

```Python
# We use a weighted dictionary search to find out where i is in
# the array.  We extrapolate position based on the largest num
# in the array and the array size and then do binary search to
# get the exact number.
if i & (i-1) == 0:  # True if i is 0 or a power of 2.
```

为了提升易读性,行注释应该至少在代码2个空格后,并以`#`后接至少1个空格开始注释部分.

另外,不要描述代码,假定阅读代码的人比你更精通Python(他只是不知道你试图做什么).

#### 3.1.6 标点,拼写和语法
注意标点,拼写和语法,写得好的注释要比写得差的好读.

注释应当是和叙事性文本一样可读,并具有合适的大小写和标点.在许多情况下,完整的句子要比破碎的句子更可读.更简短的注释如行尾的注释有时会不太正式,但是应该全篇保持风格一致.

尽管被代码审核人员指出在应该使用分号的地方使用了逗号是很令人沮丧的,将源代码维护在高度清楚可读的程度是很重要的.合适的标点,拼写和语法能够帮助达到这个目标.
## 3.2 类
如果类并非从其他基类继承而来,那么就要明确是从`object`继承而来,即便内嵌类也是如此.

**Yes:**

```Python
class SampleClass(object):
    pass

class OuterClass(object):
    class InnerClass(object):
        pass

class ChildClass(ParentClass):
    """Explicitly inherits from another class already."""
```

**No:**

```Python
class SampleClass:
    pass

class OuterClass:
    class InnerClass:
        pass
```

从`object`类继承保证了属性能够在Python2正确运行并且保护代码在Python3下出现潜在的不兼容.这样也定义了object包括`__new__`,`__init__`,`__delattr__`,`__getattribute__`,`__setattr__`,`__hash__`,`__repr__`,和`__str__`等默认特殊方法的实现.

## 3.3 字符串
使用`format`或`%`来格式化字符串,即使参数都是字符串对象,也要考虑使用`+`还是`%`及`format`.

**Yes:**

```Python
x = a + b
x = '%s, %s!' % (imperative, expletive)
x = '{}, {}'.format(first, second)
x = 'name: %s; score: %d' % (name, n)
x = 'name: {}; score: {}'.format(name, n)
x = f'name: {name}; score: {n}'  # Python 3.6+
```

**No:**

```Python
employee_table = '<table>'
for last_name, first_name in employee_list:
    employee_table += '<tr><td>%s, %s</td></tr>' % (last_name, first_name)
employee_table += '</table>'
```

避免使用`+`和`+=`操作符来在循环内累加字符串,因为字符串是不可变对象.这会造成不必要的临时变量导致运行时间以四次方增长而非线性增长.应将每个字符串都记入一个列表并使用`''.join`来将列表在循环结束后连接(或将每个子字符串写入`io.BytesIO`缓存)

**Yes:**

```Python
items = ['<table>']
for last_name, first_name in employee_list:
    items.append('<tr><td>%s, %s</td></tr>' % (last_name, first_name))
items.append('</table>')
employee_table = ''.join(items)
```

**No:**

```Python
employee_table = '<table>'
for last_name, first_name in employee_list:
    employee_table += '<tr><td>%s, %s</td></tr>' % (last_name, first_name)
employee_table += '</table>'
```

在同一个文件内,字符串引号要一致,选择`''`或者`""`并且不要改变.对于需要避免`\\`转义的时候,可以更改.

**Yes:**

```Python
Python('Why are you hiding your eyes?')
Gollum("I'm scared of lint errors.")
Narrator('"Good!" thought a happy Python reviewer.')
```

**No:**

```Python
Python("Why are you hiding your eyes?")
Gollum('The lint. It burns. It burns us.')
Gollum("Always the great lint. Watching. Watching.")
```

多行字符串多行字符串优先使用"""而非`'''`,当且只当对所有非文档字符串的多行字符串都是用`'''`而且对正常字符串都使用`'`时才可使用三单引号.docstring不论如何必须使用`"""`

多行字符串和其余代码的缩进方式不一致.如果需要避免在字符串中插入额外的空格,要么使用单行字符串连接或者带有[`textwarp.dedent()`](https://docs.python.org/3/library/textwrap.html#textwrap.dedent)的多行字符串来移除每行的起始空格.

**No:**

```Python
long_string = """This is pretty ugly.
Don't do this.
"""
```

**Yes:**

```Python
long_string = """This is fine if your use case can accept
    extraneous leading spaces."""

long_string = ("And this is fine if you can not accept\n" +
               "extraneous leading spaces.")

long_string = ("And this too is fine if you can not accept\n"
               "extraneous leading spaces.")

import textwrap

long_string = textwrap.dedent("""\
    This is also fine, because textwrap.dedent()
    will collapse common leading spaces in each line.""")
```
## 3.X 其他
- 不要在行尾加分号，也不要用分号把两行语句合并到一行
- 最大行长度是*80个字符*

    超出80字符的明确例外:
    * 长import
    * 注释中的：URL,路径,flags等
    * 不包含空格不方便分行的模块级别的长字符串常量
    * pylint的diable注释使用(如`# pylint: disable=invalid-name`)
- 括号合理使用

    尽管不必要,但是可以在元组外加括号.再返回语句或者条件语句中不要使用括号,除非是用于隐式的连接行或者指示元组.

    **Yes:**

    ```Python
    if foo:
        bar()
    while x:
        x = bar()
    if x and y:
        bar()
    if not x:
        bar()
    # For a 1 item tuple the ()s are more visually obvious than the comma.
    onesie = (foo,)
    return foo
    return spam, beans
    return (spam, beans)
    for (x, y) in dict.items(): ...
    ```

    **No:**

    ```Python
    if (x):
        bar()
    if not(x):
        bar()
    return (foo)
    ```
- 缩进用4个空格

    缩进代码段不要使用制表符,或者混用制表符和空格.如果连接多行,多行应垂直对齐,或者再次4空格缩进(这个情况下首行括号后应该不包含代码).
- 在顶级定义(函数或类)之间要间隔两行.在方法定义之间以及class所在行与第一个方法之间要空一行,def行后无空行,在函数或方法内你认为合适地方可以使用单空行.
- 括号`()`,`[]`,`{}`内部不要多余的空格.
- 于下述情况使用TODO注释: 临时的,短期的解决方案或者足够好但是不完美的解决方案.
- 变量命名
`module_name`,`package_name`,`ClassName`,`method_name`,`ExceptionName`,`function_name`,`GLOBAL_CONSTANT_NAME`,`global_var_name`,`instance_var_name`,`function_parameter_name`,`local_var_name`.

    命名函数名,变量名,文件名应该是描述性的,避免缩写,尤其避免模糊或对读者不熟悉的缩写.并且不要通过删减单词内的字母来缩短.
- 建议
    | **类型** | **公共** | **内部** |
    | --- | --- | --- |
    | 包 | `lower_with_under` |  |
    | 模块 | `lower_with_under` | `_lower_with_under` |
    | 类 | `CapWords` | `_CapWords` |
    | 异常 | `CapWords` |  |
    | 函数 | `lower_with_under()` | `_lower_with_under()` |
    | 全局/类常量 | `CAPS_WITH_UNDER` | `_CAPS_WITH_UNDER` |
    | 全局/类变量 | `lower_with_under` | `_lower_with_under` |
    | 实例变量 | `lower_with_under` | `_lower_with_under`(受保护) |
    | 方法名 | `lower_with_under()` | `_lower_with_under()`(受保护) |
    | 函数/方法参数 | `lower_with_under` |  |
    | 局部变量 | `lower_with_under` |  |
- 在Python中,`pydoc`和单元测试要求模块是可import的.所以代码在主程序执行前应进行`if __name__ == '__main__':`检查,以防止模块在import时被执行.

    ```Python
    def main():
        ...

    if __name__ == '__main__':
        main()
    ```
- 如果函数名,一直到最后的参数以及返回类型注释放在一行过长,那么分行并缩进4个空格.

# 最后
保持一致！