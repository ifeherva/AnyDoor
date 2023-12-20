import torch

old_weight = 'weight/released_step_8687.ckpt'
new_weight = 'weight/released_step_8687_fixed.ckpt'

loaded_weight = torch.load(old_weight, map_location='cpu')

to_fix = loaded_weight['state_dict']['control_model.input_hint_block.0.weight']
fixed = torch.cat([to_fix, torch.zeros(16, 1, 3, 3)], 1)
loaded_weight['state_dict']['control_model.input_hint_block.0.weight'] = fixed

torch.save(loaded_weight, new_weight)

