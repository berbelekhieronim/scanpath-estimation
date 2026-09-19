<?php
// Minimal host check. Create this file in your hosting file manager,
// open it in a browser, read the two lines, then DELETE it.
echo "PHP version: " . PHP_VERSION . "<br>";
echo "PDO drivers: " . implode(", ", PDO::getAvailableDrivers());
