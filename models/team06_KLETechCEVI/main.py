import os
import sys

# Auto-patch basicsr functional_tensor bug
try:
    import site
    for sp in site.getsitepackages():
        path = os.path.join(sp, 'basicsr/data/degradations.py')
        if os.path.exists(path):
            with open(path, 'r') as f:
                content = f.read()
            if 'functional_tensor' in content:
                content = content.replace(
                    'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
                    'from torchvision.transforms.functional import rgb_to_grayscale'
                )
                with open(path, 'w') as f:
                    f.write(content)
                print("Patched basicsr degradations.py")
except Exception as e:
    print(f"basicsr patch skipped: {e}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allforface_freq import main
__all__ = ['main']
