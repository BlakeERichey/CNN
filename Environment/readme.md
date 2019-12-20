corn_env
---
Inside this directory there are several files with `corn_env` in their name. The
word `corn_env` in all of these cases should be replaced with your desired 
environment name and they should all be the same.

This includes code snippets within the files. All instances of `corn_env` should be 
replaced with the desired name of your environment.

Learn_Corn
---
Like the `corn_env` instances there are also instances of `Learn_Corn`. These instances 
need to be replaced with your desired Class Name. This has little significance 
and will likely not be called directly by you after the package is created.

Installation
---
When all `corn_env` and `Learn_Corn` have been changed to the desired names 
run `pip install -e .` inside the Project folder to install package. This will run
the `setup.py` program and install necessary dependencies.  

This will enable you to `import corn_env` inside other projects/directories. 
`corn_env` in the import statement should be replaced with your environment name as
stated above.

For Example:
If you renamed your environment to be called `TicTacToe`, the following code 
utilize the created gym environment.  

```
import TicTacToe
env = gym.make('TicTacToe-v0')
envstate, reward, done, info = env.step(0)
env.render()
env.reset()
```