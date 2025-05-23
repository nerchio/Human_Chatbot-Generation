import tiktoken
from langchain.callbacks.base import BaseCallbackHandler

class UniversalTokenCounter(BaseCallbackHandler):
    """Universal token counter"""
    
    def __init__(self, encoding_name="cl100k_base"):
        """Initialize the counter
        
        Args:
            encoding_name: tiktoken encoding name, default is cl100k_base (GPT-4/3.5-turbo encoding)
        """
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except:
            print(f"Warning: cannot load {encoding_name} encoding, using simple character count as fallback")
            self.encoding = None
    
    def count_tokens(self, text):
        """Calculate the number of tokens in the text"""
        if self.encoding:
            # Use tiktoken to calculate
            return len(self.encoding.encode(text))
        else:
            # Fallback method: simple estimation
            # English is about 4 characters per token, Chinese is about 1-2 characters per token
            # This is a rough estimate, different models may have different results
            words = text.split()
            chinese_char_count = sum(1 for c in text if ord(c) > 127)
            english_char_count = len(text) - chinese_char_count
            
            return chinese_char_count + english_char_count // 4
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Callback when LLM starts processing"""
        # Calculate the number of tokens in the input prompts
        prompt_tokens = sum(self.count_tokens(p) for p in prompts)
        self.prompt_tokens += prompt_tokens
    
    def on_llm_end(self, response, **kwargs):
        """Callback when LLM finishes processing"""
        # If the LLM provides token counting, use it
        if hasattr(response, "llm_output") and response.llm_output is not None:
            token_usage = response.llm_output.get("token_usage", {})
            
            # Update the token count from the LLM
            if token_usage:
                # If the response contains prompt_tokens, use this data, otherwise keep our calculation
                if "prompt_tokens" in token_usage:
                    # Subtract the value we calculated before, add the actual value
                    self.prompt_tokens = self.prompt_tokens - token_usage.get("prompt_tokens", 0)
                self.completion_tokens += token_usage.get("completion_tokens", 0)
                self.total_tokens += token_usage.get("total_tokens", 0) or (token_usage.get("prompt_tokens", 0) + token_usage.get("completion_tokens", 0))
        else:
            # LLM does not provide token counting, use our own estimation
            response_text = ""
            if hasattr(response, "content"):
                response_text = response.content
            elif hasattr(response, "generations"):
                for gen in response.generations:
                    if hasattr(gen[0], "text"):
                        response_text += gen[0].text
            
            # Calculate the number of tokens in the response
            completion_tokens = self.count_tokens(response_text)
            self.completion_tokens += completion_tokens
            self.total_tokens = self.prompt_tokens + self.completion_tokens
    
    def get_stats(self):
        """Get complete statistics"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }