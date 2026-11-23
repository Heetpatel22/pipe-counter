# Link directly to the saved weights from your active training run
production_weights = '/content/runs/detect/train-5/weights/best.pt'

# 1. Test it with a known solid image path
run_production_inventory_with_type_breaker('/content/group_photo.jpg', production_weights)

# 2. Test it with a known hollow image path to watch the Type Breaker drop execution
run_production_inventory_with_type_breaker('/content/test_hollow_file.jpg', production_weights)

run_production_inventory_with_type_breaker('/content/runs/detect/solid_1.jpg', production_weights)

run_production_inventory_with_type_breaker('/content/runs/detect/solid_2.jpg', production_weights)

run_production_inventory_with_type_breaker('/content/runs/detect/solid_3.jpg', production_weights)


run_production_inventory_with_type_breaker('/content/runs/detect/hollow_7.jpg', production_weights)

run_production_inventory_with_type_breaker('/content/runs/detect/hollow_8.jpg', production_weights)
