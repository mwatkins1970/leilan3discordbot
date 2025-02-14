class UsernameNormalizer:
    def __init__(self):
        self.username_map = {}
        self.next_index = 1
        
    def normalize_username(self, original_username: str) -> str:
        """Maps an original username to a generic numbered username."""
        if original_username not in self.username_map:
            self.username_map[original_username] = f"username{self.next_index}"
            self.next_index += 1
        return self.username_map[original_username]

    def normalize_message_history(self, message_history: str) -> str:
        """
        Replace all usernames in angle brackets with normalized versions.
        Preserves the bot's name without normalization.
        """
        import re
        
        # Find all usernames in angle brackets
        username_pattern = r'<([^>]+)>'
        
        def replace_username(match):
            username = match.group(1)
            # Don't normalize the bot's name (assuming it's "Leilan" in this case)
            if username == "Leilan":
                return f"<{username}>"
            return f"<{self.normalize_username(username)}>"
            
        normalized_history = re.sub(username_pattern, replace_username, message_history)
        return normalized_history
        
    def get_username_mappings(self) -> dict:
        """Returns the current username mappings for debugging/logging."""
        return dict(self.username_map)
    

    def denormalize_message_history(self, message_history: str) -> str:
        """
        Replace all normalized usernames with their original versions
        """
        import re
        
        # Create reverse mapping
        reverse_map = {v: k for k, v in self.username_map.items()}
        
        # Find all username patterns
        username_pattern = r'<([^>]+)>'
        
        def replace_username(match):
            username = match.group(1)
            # Don't denormalize the bot's name
            if username == "Leilan":
                return f"<{username}>"
            # If it's a normalized username (e.g. username1), replace with original
            if username in reverse_map:
                return f"<{reverse_map[username]}>"
            return match.group(0)  # Keep unchanged if not found
                
        denormalized_history = re.sub(username_pattern, replace_username, message_history)
        return denormalized_history