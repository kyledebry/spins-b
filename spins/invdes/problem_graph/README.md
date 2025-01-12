Problem Graph

This is where most of the changes from the original SPINS-B code are. Adding a new feature (such as a objective) has two parts.

1) In the ``schema_*.py`` file in the optplan folder, add a node definition where you define what variables are needed to create the new item and their types
2) In the ``creator_*.py`` file, add the actual class that you are trying to create. It can either take the class from the ``schema_*.py`` file in its ``__init__`` method, or you can make a function that takes the schema class as input and returns the creator class

Then, in the ``schema_*.py`` file, you must add ``@optplan.register_node_type()`` above the class definition. In ``creator_*.py``, you add ``@optplan.register_node(schema_class_name)`` above the class, or above the creator function if you made one.

New objectives typically live in the ``*_em.py`` files. New ways of initializing the epsilon disribution live in the ``*_param.py`` files.
