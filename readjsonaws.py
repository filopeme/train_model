import json

def extract_key_value_pairs(textract_result):
    """
    Extract key-value pairs from an Amazon Textract JSON response.

    Parameters:
    textract_result (dict): The Textract JSON output (as a Python dict, e.g., loaded from a .json file).

    Returns:
    dict: A dictionary where each key is a form field name and each value is the extracted text for that field.
          If a key has no associated text, the value will be an empty string.
    """
    # Ensure we have the list of blocks
    blocks = textract_result.get('Blocks', textract_result if isinstance(textract_result, list) else [])
    
    # Dictionaries to hold blocks by Id and separate key/value blocks
    id_map = {}
    key_blocks = []        # list to store key-type blocks
    value_blocks = {}      # dict to store value-type blocks by their Id
    
    for block in blocks:
        block_id = block.get('Id')
        if block_id:
            id_map[block_id] = block  # map Id to block for quick lookup
        # Identify KEY and VALUE type blocks
        if block.get('BlockType') == 'KEY_VALUE_SET':
            entity_types = block.get('EntityTypes', [])
            if 'KEY' in entity_types:
                key_blocks.append(block)
            elif 'VALUE' in entity_types:
                value_blocks[block_id] = block

    def get_text_from_block(block):
        """
        Helper function to concatenate text from the 'WORD' children (and mark selected checkboxes).
        """
        text_parts = []
        if not block:
            return ""
        # Traverse child relationships to gather text
        for rel in block.get('Relationships', []):
            if rel.get('Type') == 'CHILD':
                for child_id in rel.get('Ids', []):
                    child = id_map.get(child_id)
                    if not child:
                        continue
                    # If the child is a word, add its text
                    if child.get('BlockType') == 'WORD':
                        text_parts.append(child.get('Text', ''))
                    # If the child is a selection element (e.g., a checkbox), mark it if selected
                    elif child.get('BlockType') == 'SELECTION_ELEMENT':
                        if child.get('SelectionStatus') == 'SELECTED':
                            text_parts.append("X")  # using "X" to denote a checked box
        # Join all text parts with space and strip extra whitespace
        return " ".join(text_parts).strip()

    # Extract key-value pairs
    kv_pairs = {}
    for key_block in key_blocks:
        key_text = get_text_from_block(key_block)
        if not key_text:
            continue  # skip if no text found for the key
        value_text = ""
        # Find the associated value block via the VALUE type relationship
        for rel in key_block.get('Relationships', []):
            if rel.get('Type') == 'VALUE':
                for value_id in rel.get('Ids', []):
                    value_block = value_blocks.get(value_id)
                    if value_block:
                        value_text = get_text_from_block(value_block)
                        break
            if value_text:
                break  # exit if value found
        kv_pairs[key_text] = value_text

    return kv_pairs

# Example usage:
#if __name__ == "__main__":
#    # Load Textract JSON output from a file (replace 'textract_output.json' with your filename)
#    with open("textract_output.json", "r") as f:
#        textract_data = json.load(f)
#    # Extract key-value pairs
#    result = extract_key_value_pairs(textract_data)
#    # Print the results
#    for key, val in result.items():
#        print(f"{key}: {val}")
