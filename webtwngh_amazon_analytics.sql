-- phpMyAdmin SQL Dump
-- version 5.2.2
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3306
-- Generation Time: Oct 18, 2025 at 09:51 PM
-- Server version: 11.4.8-MariaDB-cll-lve-log
-- PHP Version: 8.3.26

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `webtwngh_amazon_analytics`
--

-- --------------------------------------------------------

--
-- Table structure for table `customers`
--

CREATE TABLE `customers` (
  `customer_id` varchar(128) NOT NULL,
  `customer_city` varchar(128) DEFAULT NULL,
  `customer_state` varchar(128) DEFAULT NULL,
  `age_group` varchar(32) DEFAULT NULL,
  `customer_spending_tier` varchar(32) DEFAULT NULL,
  `is_prime_member` tinyint(1) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `products`
--

CREATE TABLE `products` (
  `product_id` varchar(128) NOT NULL,
  `product_name` varchar(512) DEFAULT NULL,
  `category` varchar(128) DEFAULT NULL,
  `subcategory` varchar(128) DEFAULT NULL,
  `brand` varchar(128) DEFAULT NULL,
  `base_price_2015` double DEFAULT NULL,
  `is_prime_eligible` tinyint(1) DEFAULT NULL,
  `launch_year` int(11) DEFAULT NULL,
  `weight_kg` double DEFAULT NULL,
  `model` varchar(256) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `time_dimension`
--

CREATE TABLE `time_dimension` (
  `date_key` date NOT NULL,
  `year` int(11) DEFAULT NULL,
  `quarter` int(11) DEFAULT NULL,
  `month` int(11) DEFAULT NULL,
  `day` int(11) DEFAULT NULL,
  `month_name` varchar(20) DEFAULT NULL,
  `quarter_name` varchar(6) DEFAULT NULL,
  `is_weekend` tinyint(1) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

-- --------------------------------------------------------

--
-- Table structure for table `transactions`
--

CREATE TABLE `transactions` (
  `transaction_id` varchar(128) NOT NULL,
  `customer_id` varchar(128) DEFAULT NULL,
  `product_id` varchar(128) DEFAULT NULL,
  `order_date` date DEFAULT NULL,
  `order_month` int(11) DEFAULT NULL,
  `order_year` int(11) DEFAULT NULL,
  `order_quarter` int(11) DEFAULT NULL,
  `product_name` varchar(512) DEFAULT NULL,
  `category` varchar(128) DEFAULT NULL,
  `subcategory` varchar(128) DEFAULT NULL,
  `brand` varchar(128) DEFAULT NULL,
  `product_rating` double DEFAULT NULL,
  `original_price_inr` double DEFAULT NULL,
  `discount_percent` double DEFAULT NULL,
  `final_amount_inr` double DEFAULT NULL,
  `delivery_charges` double DEFAULT NULL,
  `delivery_days` int(11) DEFAULT NULL,
  `payment_method` varchar(64) DEFAULT NULL,
  `is_prime_member` tinyint(1) DEFAULT NULL,
  `is_festival_sale` tinyint(1) DEFAULT NULL,
  `return_status` varchar(64) DEFAULT NULL,
  `customer_rating` double DEFAULT NULL,
  `customer_city` varchar(128) DEFAULT NULL,
  `customer_state` varchar(128) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `customers`
--
ALTER TABLE `customers`
  ADD PRIMARY KEY (`customer_id`);

--
-- Indexes for table `products`
--
ALTER TABLE `products`
  ADD PRIMARY KEY (`product_id`);

--
-- Indexes for table `time_dimension`
--
ALTER TABLE `time_dimension`
  ADD PRIMARY KEY (`date_key`);

--
-- Indexes for table `transactions`
--
ALTER TABLE `transactions`
  ADD PRIMARY KEY (`transaction_id`),
  ADD KEY `idx_order_date` (`order_date`),
  ADD KEY `idx_customer_id` (`customer_id`),
  ADD KEY `idx_product_id` (`product_id`),
  ADD KEY `idx_category` (`category`),
  ADD KEY `idx_year_month` (`order_year`,`order_month`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
