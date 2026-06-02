```markdown
# Apohara_Context_Forge Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and conventions used in the Apohara_Context_Forge Python repository. You'll learn how to structure files, write imports and exports, follow commit message standards, and organize tests. These patterns ensure consistency, readability, and maintainability across the codebase.

## Coding Conventions

### File Naming
- Use **snake_case** for all file names.
  - Example: `context_manager.py`, `data_processor.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .utils import parse_context
    from .models import ContextModel
    ```

### Export Style
- Use **named exports** by specifying `__all__` in modules.
  - Example:
    ```python
    __all__ = ['ContextManager', 'parse_context']
    ```

### Commit Messages
- Follow the **Conventional Commits** format.
- Use the `feat` prefix for new features.
- Keep commit messages concise (average ~51 characters).
  - Example:
    ```
    feat: add context parsing for user input
    ```

## Workflows

### Adding a New Feature
**Trigger:** When implementing a new capability or module  
**Command:** `/add-feature`

1. Create a new Python file using snake_case (e.g., `new_feature.py`).
2. Implement the feature using relative imports as needed.
3. Add named exports via `__all__`.
4. Write or update corresponding test files (`new_feature.test.py`).
5. Commit changes with a message like:  
   `feat: brief description of the new feature`
6. Push your branch and open a pull request.

### Writing Tests
**Trigger:** When adding or updating code  
**Command:** `/write-test`

1. Create a test file named with the pattern `*.test.py` (e.g., `context_manager.test.py`).
2. Write test cases for new or modified functions/classes.
3. Use the project's preferred (unknown) testing framework.
4. Run tests locally to ensure they pass.
5. Commit test files with a descriptive message.

## Testing Patterns

- Test files follow the pattern: `*.test.py`
- Each test file is placed alongside or within the relevant module directory.
- The specific testing framework is not specified, but standard Python testing practices apply.
- Example test file:
  ```python
  # context_manager.test.py

  def test_parse_context():
      result = parse_context("example input")
      assert result == expected_output
  ```

## Commands
| Command      | Purpose                                    |
|--------------|--------------------------------------------|
| /add-feature | Start the workflow for adding a new feature|
| /write-test  | Guide for writing and running tests        |
```
