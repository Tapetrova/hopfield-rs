use hopfield;

fn main() {
    hopfield::thermal_low_load_experiment();

    // Запуск исправленного MNIST експеримента
    // hopfield::mnist_projection_experiment();

    // Запуск експеримента с рекордными замерами
    // hopfield::record_capacity_experiment();

    // Запуск MNIST експеримента(старая версия, поломанная)
    // hopfield::mnist_experiment();

    // Запуск експеримента емкости
    // hopfield::capacity_experiment();

}
