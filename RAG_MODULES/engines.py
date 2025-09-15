from baseclass import VectorEngine
import numpy as np
import requests
from llm import LLM

"""-------------------------Uses Ollama's "nomic-embed-text" model for embeddings and follows semantic chunking strategy---------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

class NomicEngine(VectorEngine):
    def __init__(self):
        self.llm_obj = LLM()
        self.model = "nomic-embed-text"

    def get_embeddings(self, text: str) -> np.ndarray:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            }
        )
        response.raise_for_status()
        return np.array(response.json()["embedding"], dtype=np.float32)

    def re_rank(self, query: str, chunks: list[str]) -> list[float]:
        pass

    def chunk_text(self,text:str) -> list[str]:
        """
        Chunk text into topic-based blocks using LLM-based topic detection with overlap.
        Strategy:
        1. Start with 128-word blocks
        2. Ask LLM if there's a second topic in the block
        3. If no second topic found, add next 128-word block and ask again
        4. If second topic found, prepend 10% of previous chunk to current chunk
        5. First part becomes finalized chunk, second part prepended to next block
        6. Continue recursively until entire document is processed
        """
        def split_into_words(text):
            """Split text into words for counting"""
            return text.split()
        
        def join_words(words):
            """Join words back into text"""
            return ' '.join(words)
        
        def get_second_topic_part(block_text):
            """Ask LLM if there's a second topic and return the second part if found"""
            prompt = f"""
            Analyze this text block and determine if it contains a second distinct topic.
            
            Text block:
            {block_text}
            
            If there is a second topic in this block, return ONLY the text from where the second topic begins (including that sentence).
            If there is only one topic throughout the block, return "NO_SECOND_TOPIC".
            
            Be precise and only return the actual text of the second topic part, or "NO_SECOND_TOPIC".
            """
            
            response = self.llm_obj.get_openai_response(prompt)
            
            # Check if LLM found a second topic
            if response.strip() == "NO_SECOND_TOPIC":
                return None
            else:
                # Return the second topic part
                return response.strip()
        
        chunks = []
        words = split_into_words(text)
        current_position = 0
        previous_chunk = None
        
        while current_position < len(words):
            # Start with 128-word block
            block_size = 128
            accumulated_words = []
            
            # Keep adding blocks until LLM detects a second topic
            while current_position < len(words):
                # Get next block of words
                end_position = min(current_position + block_size, len(words))
                current_block_words = words[current_position:end_position]
                accumulated_words.extend(current_block_words)
                
                # Create accumulated text for LLM analysis
                accumulated_text = join_words(accumulated_words)
                
                # Ask LLM if there's a second topic in the accumulated text
                second_topic_part = get_second_topic_part(accumulated_text)
                
                if second_topic_part is None:
                    # No second topic found, continue adding more blocks
                    current_position = end_position
                    
                    # If we've reached the end of the text, this becomes the final chunk
                    if current_position >= len(words):
                        # Create final chunk with overlap from previous chunk
                        if previous_chunk is not None:
                            prev_words = split_into_words(previous_chunk)
                            overlap_size = max(1, int(len(prev_words) * 0.1))
                            overlap_words = prev_words[-overlap_size:]
                            final_chunk_text = join_words(overlap_words + accumulated_words)
                        else:
                            final_chunk_text = accumulated_text
                        
                        chunks.append(final_chunk_text)
                        break
                else:
                    # Second topic found, need to split
                    # Find where the second topic starts in the accumulated text
                    second_topic_words = split_into_words(second_topic_part)
                    
                    # Find the position where second topic starts
                    first_part_words = []
                    for i, word in enumerate(accumulated_words):
                        # Check if remaining words match the start of second topic
                        remaining_words = accumulated_words[i:]
                        if len(remaining_words) >= len(second_topic_words):
                            # Check if the remaining words start with second topic words
                            if remaining_words[:len(second_topic_words)] == second_topic_words:
                                first_part_words = accumulated_words[:i]
                                break
                    
                    # If we couldn't find the split point, use a fallback
                    if not first_part_words:
                        # Fallback: split at roughly 75% of the accumulated text
                        split_point = int(len(accumulated_words) * 0.75)
                        first_part_words = accumulated_words[:split_point]
                        second_topic_words = accumulated_words[split_point:]
                    
                    # Create the first part chunk with overlap from previous chunk
                    if first_part_words:
                        first_part_text = join_words(first_part_words)
                        
                        # If we have a previous chunk, prepend 10% of it
                        if previous_chunk is not None:
                            prev_words = split_into_words(previous_chunk)
                            overlap_size = max(1, int(len(prev_words) * 0.1))  # 10% overlap, minimum 1 word
                            overlap_words = prev_words[-overlap_size:]
                            final_chunk_text = join_words(overlap_words + first_part_words)
                        else:
                            final_chunk_text = first_part_text
                        
                        chunks.append(final_chunk_text)
                        previous_chunk = final_chunk_text
                    
                    # Prepend the second part to the next iteration
                    # Move position to where first part ended
                    current_position += len(first_part_words)
                    
                    # If we're at the end, add the remaining part as the last chunk
                    if current_position >= len(words):
                        if second_topic_words:
                            second_part_text = join_words(second_topic_words)
                            
                            # Add overlap from previous chunk if available
                            if previous_chunk is not None:
                                prev_words = split_into_words(previous_chunk)
                                overlap_size = max(1, int(len(prev_words) * 0.1))
                                overlap_words = prev_words[-overlap_size:]
                                final_chunk_text = join_words(overlap_words + second_topic_words)
                            else:
                                final_chunk_text = second_part_text
                            
                            chunks.append(final_chunk_text)
                        break
                    
                    # Break out of the inner while loop to start fresh with next block
                    break
        
        return chunks
"""------------------------------------------------------------------------------------------------"""