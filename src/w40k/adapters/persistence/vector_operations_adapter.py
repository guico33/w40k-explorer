"""VectorOperations adapter implementing the VectorOperationsPort."""

from ...infrastructure.database.vector_operations import VectorOperations


class VectorOperationsAdapter(VectorOperations):
    """Adapter that wraps existing VectorOperations to implement the port interface.

    This adapter allows the existing VectorOperations class to be used through
    the VectorOperationsPort protocol interface, enabling dependency injection.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the adapter by passing all arguments to the parent class."""
        super().__init__(*args, **kwargs)

    # The parent class already implements the required methods:
    # - search_similar_chunks()
    # - get_embedding_stats()

    # No additional implementation needed - the adapter pattern
    # allows us to use existing functionality while conforming
    # to the VectorOperationsPort protocol interface.
