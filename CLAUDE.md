# Running tests

This project uses `uv` to manage python dependencies.

DO:
- Use `uv run` to run scripts
- Use `uv run --extra dev pytest` to run tests

DON'T:
- Use `python3` invocations directly
- Set up your own virtualenvs

# Important workflow instructions

## jj new -m ...

This project uses jj (jujutsu) to manage version control.
In jj, ALL changes are automatically committed: there is no staging area.
That means that you should run `jj new -m "my message..."` BEFORE starting a new coding task.

### When to run `jj new -m "..."`
- At the *start* of each new task (not after completing it)
- When switching to a different logical unit of work
- Before beginning any subtask that could stand alone as a commit
- Whenever you notice you're doing "one more thing"

### Common commands
- `jj new -m "description"` - Creates a new revision
- `jj status` - check current status
- `jj log` - check revision history

## Task tracking

We use `tk` for task tracking. Rather than using your TodoWrite tool, you should prefer to use `tk create`
or the corresponding skill to create a task in the tracker. This lets us persist tasks across sessions.

### Workflow:
- Create a ticket or multiple tickets to track work
- See what work is ready to start: `tk ready`
- Claim work with `tk start <id>`
- Mark a ticket as finished with `tk close`

### When to create a tk ticket
- When formulating a task list, file them as tickets.
- Whenever something is going to take more than a few minutes, file it
- Whenever a task breaks down into 

Run `tk help` for detailed help.


